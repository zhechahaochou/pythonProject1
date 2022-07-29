import numpy as np
import torch
from torch.utils.data import  DataLoader
from torch.utils.data import  Dataset
from sklearn.model_selection import train_test_split

fp = '/Users/mac/Documents/diabetes.csv.gz'
raw_data = np.loadtxt(fp, delimiter=',', dtype=np.float32)
X = raw_data[:, :-1]
y = raw_data[:, [-1]]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)
Xtest = torch.from_numpy(Xtest)
Ytest = torch.from_numpy(Ytest)

# prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]  # 获取行数
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

train_dataset = DiabetesDataset(Xtrain, Ytrain)
# dataset = DiabetesDataset(fp)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=1)



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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

'''新！！！'''
def train(epoch):
    train_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)

        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        count = i

    if epoch % 2000 == 1999:
        print("train loss:", train_loss / count, end=',')
def test():
    with torch.no_grad():
        y_pred = model(Xtest)
        y_pred_label = torch.where(y_pred>=0.5,torch.tensor([1.0]),torch.tensor([0.0]))
        acc = torch.eq(y_pred_label, Ytest).sum().item() / Ytest.size(0)
        print("test acc:", acc)

'''新的！！！封装'''
if __name__ == '__main__':
    for epoch in range(50000):
        train(epoch)
        if epoch%20000==1999:
            test()




