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
# 删除列
data.drop(['total cost'], axis=1, inplace=True)

# 保存修改后的数据
data.to_csv('../data/New_Train_Data.csv', index=False)
selected_columns = data.columns.tolist()
selected_columns.remove('cost rank')
print(selected_columns)

# 读取测试数据并保存为所需格式
test_data = pd.read_csv('../data/Test_Data.csv')
new_test_data = test_data[selected_columns]
new_test_data['cost rank'] = ''
new_test_data.to_csv('../data/New_Test_Data.csv', index=False)
