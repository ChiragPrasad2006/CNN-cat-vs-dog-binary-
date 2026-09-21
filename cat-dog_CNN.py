import os
from pathlib import Path
import torchmetrics
from torch import device
import torch
import torchvision
from torch import nn
from torchmetrics import Accuracy
from torch.utils.data import TensorDataset, DataLoader
from timeit import default_timer as timer
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def preprocess_images(src_dir="data", dst_dir="data_224", size=(224, 224)):
    if os.path.exists(dst_dir) and len(os.listdir(dst_dir)) > 0:
        return
    print(f"Pre-resizing dataset to {size} in '{dst_dir}' (done once)...")
    os.makedirs(dst_dir, exist_ok=True)
    for class_name in ["Cat", "Dog"]:
        src_class_dir = os.path.join(src_dir, class_name)
        dst_class_dir = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        for img_name in tqdm(os.listdir(src_class_dir), desc=f"Resizing {class_name}"):
            src_path = os.path.join(src_class_dir, img_name)
            dst_path = os.path.join(dst_class_dir, img_name)
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB").resize(size)
                    img.save(dst_path, "JPEG")
            except Exception:
                continue

def load_dataset_into_memory(root="data_224"):
    print(f"Caching dataset from '{root}' into RAM...")
    classes = ["Cat", "Dog"]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    images = []
    labels = []
    
    for class_name in classes:
        class_dir = os.path.join(root, class_name)
        cls_idx = class_to_idx[class_name]
        for img_name in tqdm(os.listdir(class_dir), desc=f"Loading {class_name} into RAM"):
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img_tensor = torchvision.transforms.functional.pil_to_tensor(img.convert("RGB"))
                    images.append(img_tensor)
                    labels.append(cls_idx)
            except Exception:
                continue
                
    all_images = torch.stack(images)  # uint8 tensor: ~3.7 GB RAM for 224x224
    all_labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(all_images, all_labels), class_to_idx

# GPU-accelerated real-time training augmentations
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(degrees=15),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

class CatDogCNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Modern classifier head: Global Average Pooling + Dropout
        self.block4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

def print_train_test_time(start: float, end: float, device=None):
    total_time = end - start 
    print(f"train time on {device}: {total_time:.3f} seconds")
    return total_time

def train_model(model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, train_dataloader: DataLoader, device: torch.device):
    model.train()
    train_loss, samples_processed = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device, non_blocking=True).float() / 255.0
        X = train_transform(X)  # Real-time GPU data augmentation
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        samples_processed += len(X)
        if (batch + 1) % 1000 == 0:
            print(f"Looked at:{samples_processed}/{len(train_dataloader.dataset)}")
    train_loss /= len(train_dataloader)
    return train_loss

def test_model(model: torch.nn.Module, test_dataloader: DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test = X_test.to(device, non_blocking=True).float() / 255.0
            y_test = y_test.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                test_pred = model(X_test)
                loss = loss_fn(test_pred, y_test)
            
            test_loss += loss.item()
            acc = accuracy_fn(test_pred.argmax(dim=1), y_test)
            test_acc += acc.item()
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"\ntest_loss:{test_loss}, test_accuracy:{test_acc}")
    return test_loss, test_acc

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Optimize cuDNN algorithm selection for static shapes
    torch.backends.cudnn.benchmark = True

    # Pre-resize images once on disk to 224x224
    preprocess_images(src_dir="data", dst_dir="data_224", size=(224, 224))

    # Load 224x224 dataset into RAM once (~3.7 GB RAM)
    full_dataset, class_to_idx = load_dataset_into_memory(root="data_224")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {class_to_idx}")

    # split dataset
    torch.manual_seed(0)
    train_size = int(0.8 * len(full_dataset))
    test_size = int(len(full_dataset) - train_size)
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # batch_size=64 fits comfortably in 8GB VRAM with 224x224 images
    train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    # model
    model_CNN_Cat_Dog = CatDogCNN(input_shape=3, hidden_units=128, output_shape=2).to(device)

    # AdamW optimizer with Learning Rate Scheduler
    optimizer = torch.optim.AdamW(params=model_CNN_Cat_Dog.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler('cuda')
    
    # loss function & accuracy metric
    loss_fn = nn.CrossEntropyLoss().to(device)
    acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

    print_train_test_time_start = timer()

    EPOCHS = 100
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_model(model=model_CNN_Cat_Dog, loss_fn=loss_fn, optimizer=optimizer, scaler=scaler, train_dataloader=train_dataloader, device=device)
        test_loss, test_acc = test_model(model=model_CNN_Cat_Dog, test_dataloader=test_dataloader, loss_fn=loss_fn, accuracy_fn=acc_fn, device=device)
        
        # Step scheduler based on validation loss
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    print_train_test_time_end = timer()
    total_time = print_train_test_time(start=print_train_test_time_start, end=print_train_test_time_end, device=device)

    # Save trained model
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "CNN_cat_dog_binary.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_CNN_Cat_Dog.state_dict(), f=MODEL_SAVE_PATH)
