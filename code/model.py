import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the data
data = pd.read_csv("../data/New_Train_Data.csv")


# Preprocess the data
X = data.iloc[:, :-1].values
y = data["cost rank"].values - 1  # transfer the label to 0-3

# divide the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#Scale or normalize the numerical range of input features
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dataset && DataLoader
class HouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HouseDataset(X_train, y_train)
test_dataset = HouseDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# PyTorch
class Net(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.5, l2_reg=0.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, output_size)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.dropout1(Swish()(self.fc1(x)))
        x = self.dropout2(Swish()(self.fc2(x)))
        x = Swish()(self.fc3(x))
        x = self.fc4(x)
        return x

    def l2_regularization(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)**2
        return self.l2_reg * l2_loss

# 初始化模型、损失函数和优化器
input_size = X.shape[1]
output_size = len(np.unique(y))
model = Net(input_size, output_size)
criterion = nn.CrossEntropyLoss()
## Using Adam algorithm to optimize parameters
## weight_decay to adjust the optimize value to certain parameters
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) 

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.1)

## Train the model
num_epochs = 100
loss_history = []
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y) + model.l2_regularization()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    ## calculate the loss rate every epoch
    epoch_loss /= num_batches
    loss_history.append(epoch_loss)
    
    scheduler.step()
    
    ## Evaluate the model on validation set and record the loss
    ## Adjust the learning rate based on validation loss

# 绘制训练损失曲线
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

## Evaluate the model
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
with torch.no_grad():
    test_output = model(X_test_tensor)
    _, predicted = torch.max(test_output.data, 1)


## calculate Accuracy, Precision, Recall and F1 score
## Using average = 'weighted' to consider the nums of different categories
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average='weighted')
recall = recall_score(y_test, predicted, average='weighted')
f1 = f1_score(y_test, predicted, average='weighted')
## print the rates:
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))