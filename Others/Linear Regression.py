import torch
'''prepare dataset'''
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.tensor([[[1.0], [2.0], [3.0]]])
y_data = torch.tensor([[[2.0], [4.0], [6.0]]])
'''design model using class'''
class LinearModel(torch.nn.Module): #所有模型都要继承自torch.nn.Module
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1,1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

'''构造损失函数和优化器construct loss and optimizer'''
criterion = torch.nn.MSELoss(size_average=False)
#随机梯度下降，批量梯度下降，还是mini-batch梯度下降，用的API都是 torch.optim.SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#lr learn rate学习率
'''训练过程training cycle'''
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()#梯度归零
    loss.backward()#反向传播，计算梯度
    optimizer.step()#权重更新，w.b的值
#
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
#test model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
