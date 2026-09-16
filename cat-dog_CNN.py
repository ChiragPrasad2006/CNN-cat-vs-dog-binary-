import torchmetrics
from torch import device
import torch
import torchvision
from torch import nn
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from timeit import default_timer as timer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(0)
class CatDogCNN(nn.Module):
    def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block4=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        return x

#Transform dataset/define Transformer size and convert to tensor    
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((128,128)),
                            torchvision.transforms.ToTensor()]
)

#load dataset
full_dataset=ImageFolder(root="data",transform=transform)
print(len(full_dataset))
print(full_dataset.class_to_idx)

#split dataset
train_size=int(0.8*len(full_dataset))
test_size=int(len(full_dataset)-train_size)
train_data,test_data=torch.utils.data.random_split(full_dataset,[train_size,test_size],generator=torch.Generator().manual_seed(0))
print(len(train_data))
print(len(test_data))

#wrap into dataloaders
train_dataloader=torch.utils.data.DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_dataloader=torch.utils.data.DataLoader(dataset=test_data,batch_size=32,shuffle=False)

#model
model_CNN_Cat_Dog=CatDogCNN(input_shape=3,hidden_units=128,output_shape=2).to(device)
torch.manual_seed(0)

#optimizer
optimizer=torch.optim.Adam(params=model_CNN_Cat_Dog.parameters(),lr=0.001)
#loss function
loss_fn=nn.CrossEntropyLoss().to(device)
#accuracy_fn
acc_fn=Accuracy(task="multiclass",num_classes=2).to(device)

def print_train_test_time(start: float, end: float, device=None):
    total_time = end - start 
    print(f"train time on {device}: {total_time:.3f} seconds")
    return total_time

print_train_test_time_start = timer()

def train_model(model:torch.nn.Module,loss_fn:torch.nn.Module,optimizer:torch.optim,train_dataloader:torch.utils.data.DataLoader,device:torch.device):
    model.train()
    train_loss,samples_processed=0,0
    for batch,(X,y) in enumerate(train_dataloader):
        X,y=X.to(device),y.to(device)
        y_pred=model(X)
        loss=loss_fn(y_pred,y)
        train_loss=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        samples_processed+=len(X)
        if(batch+1)%1000==0:
            print(f"Looked at:{samples_processed}/{len(train_dataloader.dataset)}")
    train_loss /=len(train_dataloader)
    return train_loss

def test_model(model:torch.nn.Module,test_dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,accuracy_fn,device:torch.device):
    test_loss,test_acc=0,0
    model.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:
            test_pred=model(X_test)
            loss=loss_fn(test_pred,y_test)
            test_loss+=loss.item()
            acc=acc_fn(test_pred.argmax(dim=1),y_test)
            test_acc+=acc.item()
        test_loss/=len(test_dataloader)
        test_acc/=len(test_dataloader)
    print(f"\ntest_loss:{test_loss},test_accuracy:{test_acc}")

EPOCHS=100
for epoch in tqdm(range(EPOCHS)):
    train_loss = train_model(model_CNN_Cat_Dog, train_dataloader, loss_fn, optimizer, device)
    test_loss, test_acc = test_model(model_CNN_Cat_Dog, test_dataloader, loss_fn, acc_fn, device)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

print_train_test_time_end = timer()
total_time = print_train_test_time(start=print_train_test_time_start, end=print_train_test_time_end, device=device)



