import torch
import matplotlib.pyplot as plt

print("sd"+torch.version.cuda) 

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [5.0]])

class LinearModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
model = LinearModel()

criterion:torch.nn.MSELoss = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []
for epoch in range(1000):
    epoch_list.append(epoch)
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    loss_list.append(loss.item())
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print(y_test)
plt.plot(epoch_list, loss_list, scaley=False)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()