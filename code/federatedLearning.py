"""
Created by Charles-Deng
870055485@qq.com
Date: 2023/4/7 12:08
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from machineLearning import HouseDataset, Net, preprocess_data, evaluate_fold, train_model

def average_weights(global_model, client_models):
    """
    计算客户端模型权重的平均值，并更新全局模型。
    """
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.zeros_like(global_dict[k])

    for client_model in client_models:
        client_dict = client_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] += client_dict[k]

    for k in global_dict.keys():
        global_dict[k] = global_dict[k] / len(client_models)

    global_model.load_state_dict(global_dict)

def federated_learning_round(global_model, X, y, num_clients=4, num_epochs=10):
    """
    执行一轮联邦学习。
    """
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    client_data = [(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)]

    for client_idx, (train_idx, test_idx) in enumerate(client_data):
        print(f"Client {client_idx + 1}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_dataset = HouseDataset(X_train, y_train)
        test_dataset = HouseDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = client_models[client_idx]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.1)

        losses, val_losses = train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs)

    average_weights(global_model, client_models)

def main():

    data = pd.read_csv("../data/2.4_Train_Data_New.csv")
    X, y = preprocess_data(data)

    input_size = X.shape[1]
    output_size = len(np.unique(y))
    global_model = Net(input_size, output_size)

    # Split data into global training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    num_rounds = 5
    for round_idx in range(num_rounds):
        print(f"Federated Learning Round {round_idx + 1}")
        federated_learning_round(global_model, X_train, y_train)

    # Evaluate global model on testing set
    test_dataset = HouseDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    global_model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = global_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.numpy())
            predictions.extend(predicted.numpy())

    accuracy, precision, recall, f1 = evaluate_fold(true_labels, predictions)
    print(f"Global Model Accuracy: {accuracy:.4f}")
    print(f"Global Model Precision: {precision:.4f}")
    print(f"Global Model Recall: {recall:.4f}")
    print(f"Global Model F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()

