import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import  DataLoader
from torch.utils.data import  Dataset

#prepare dataset
'''新的！！！'''
class DiabetesDataset(Dataset):
    def __init__(self, fp):
        xy = np.loadtxt(fp, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]#获取行数
        self.x_data = torch.from_numpy(xy[:, :-1])  # 行取全部，列不要最后一列
        self.y_data = torch.from_numpy(xy[:, [-1]])  # 行取全部，列只要最后一列

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

fp = '/Users/mac/Documents/diabetes.csv.gz'
dataset = DiabetesDataset(fp)
train_loader = DataLoader(dataset = dataset, batch_size=32, shuffle=True, num_workers=2)



# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

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

# epoch_list = []#新内容！！！写这个是为了输出loss和epoch的图
# loss_list = []
# training cycle forward, backward, update
'''新的！！！封装'''
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):# train_loader 是先shuffle后mini_batch
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()




