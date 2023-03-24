import pandas as pd

# 读取数据
data = pd.read_csv('../data/Correlation_Data_Sumed.csv')

# 定义分类函数
def classify_total_cost(total_cost):
    if total_cost >= 0 and total_cost < 300000:
        return 1
    elif total_cost >= 300000 and total_cost < 500000:
        return 2
    elif total_cost >= 500000 and total_cost < 700000:
        return 3
    else:
        return 4

# 应用分类函数并添加新列
data['cost rank'] = data['total cost'].apply(lambda x: classify_total_cost(x))

# 保存修改后的数据
data.to_csv('../data/New_Train_Data.csv', index=False)