import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset=datasets.MNIST(root='../dataset/mnist', train=True, transform=transforms, download=True)
test_dataset=datasets.MNIST(root='../dataset/mnist', train=False, transform=transforms, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, padding=2) #28->28
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5) #28->24
        self.conv3 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1) 
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(32, 24)
        self.fc2 = torch.nn.Linear(24, 10)
        self.fc3 = torch.nn.Linear(8, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x))) #4,28,28->4,14,14
        #print(x.shape)
        x = F.relu(self.pooling(self.conv2(x))) #4,14,14->12,6,6
        #print(x.shape)
        x = F.relu(self.pooling(self.conv3(x))) #12,6,6->2,4,4
        #print(x.shape)
        x = x.view(batch_size, -1)
        #print(x.shape)
        x = F.relu(self.fc2(self.fc(x)))
        #x = F.relu(self.fc2(x))
        #x = self.fc2(x)
        #x = self.fc3(x)
        return x
    
model = Net()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        #print(inputs.shape)
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss=0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set : %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()