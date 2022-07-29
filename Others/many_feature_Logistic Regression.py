import matplotlib.pyplot as plt
import numpy as np
import torch
fp = '/Users/mac/Documents/diabetes.csv.gz'
# xy = pd.read_csv(fp, dtype=np.float32)
xy = np.loadtxt(fp, delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])#行取全部，列不要最后一列
y_data = torch.from_numpy(xy[:,[-1]])#行取全部，列只要最后一列
# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()#新的！！将其看作是网络的一层，而不是简单的函数使用

    def forward(self, x):
        # y_pred = torch.sigmoid(self.linear(x))
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))#y_pred(y hat)
        return x


model = Model()

# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
# criterion = torch.nn.BCELoss(size_average=False)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []#新内容！！！写这个是为了输出loss和epoch的图
loss_list = []
# training cycle forward, backward, update
for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())
    epoch_list.append(epoch)#新内容！！！
    loss_list.append(loss.item())#新内容！！！

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# print('w = ', model.linear.weight.item())
# print('b = ', model.linear.bias.item())
#
# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
# print('y_pred = ', y_test.data)
