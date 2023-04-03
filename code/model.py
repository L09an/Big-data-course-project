import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Read the data
data = pd.read_csv("../data/New_Train_Data.csv")

# Preprocess the data
# Apply one-hot encoding to categorical features
categorical_features = ['zip code', 'city']
numerical_features = [col for col in data.columns if col not in categorical_features + ["cost rank"]]

# Create a preprocessor that can handle both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(with_mean=False), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X = preprocessor.fit_transform(data.iloc[:, :-1])
y = data["cost rank"].values - 1  # transfer the label to 0-3

# Dataset && DataLoader
class HouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)  # Convert sparse matrix to dense
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# PyTorch
class Net(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.5, l2_reg=0.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_prob)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(dropout_prob)
        
        self.fc5 = nn.Linear(32, output_size)

        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
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

# Training the model
num_splits = 5
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

fold_accuracies = []
fold_precisions = []
fold_recalls = []

losses_per_fold = []
val_losses_per_fold = []

fold_class_reports = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_dataset = HouseDataset(X_train, y_train)
    test_dataset = HouseDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Net(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.1)

    num_epochs = 60
    losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels) + model.l2_regularization()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    losses_per_fold.append(losses)
    val_losses_per_fold.append(val_losses)

    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    report = classification_report(y_true, y_pred, output_dict=True)
    fold_class_reports.append(report)
    
# Plotting the metrics for each fold
num_classes = len(np.unique(y))
metrics = ['precision', 'recall', 'f1-score']

for i, report in enumerate(fold_class_reports):
    plt.figure()
    
    for j, metric in enumerate(metrics):
        class_metrics = [report[str(c)][metric] for c in range(num_classes)]
        positions = [x + j * 0.2 for x in range(num_classes)]
        plt.bar(positions, class_metrics, width=0.2, label=f'{metric} - Fold {i + 1}')
    
    plt.xticks(range(num_classes), [f'Class {c}' for c in range(num_classes)])
    plt.xlabel('Classes')
    plt.ylabel('Metric Value')
    plt.title(f'Fold {i + 1} Metrics')
    plt.legend()
    plt.show()