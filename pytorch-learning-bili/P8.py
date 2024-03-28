import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets
import numpy as np
import os
#print(train_dataset.data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
class DiabetesDataset(Dataset):
    def __init__(self, filepath) -> None:
        super().__init__()
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        #print(filepath)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        #print(self.len)
        return self.len

dataset = DiabetesDataset("diabetes.csv.gz")
train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=2)

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x
    
model = Model()
model.to(device)

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



def train(epoch):
    for i,data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())

        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        