import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# 读取 Train_Data.csv 文件
data = pd.read_csv('../data/Train_Data_sumed.csv')
data.drop(['unit price of residence space', 'unit price of building space'], axis=1, inplace=True)

# 保存为新的csv文件
data.to_csv('../data/Correlation_Data.csv', index=False)

# 读取 Correlation_Data.csv 文件
data = pd.read_csv('../data/Correlation_Data.csv')

# 将列date和zip code转化为秩次
data['date'] = rankdata(data['date'])
data['zip code'] = rankdata(data['zip code'])

# 计算各变量和 total cost 的相关性
corr = data.corr(method = "spearman")['total cost']

# 筛选相关性绝对值大于等于 0.35 的变量
significant_vars = corr[(abs(corr) >= 0.35) & (corr.index != 'total cost')].index.tolist()

# 打印初步的筛选结果
print(significant_vars)
 
# 找出significant_vars以外的，与significant_vars中任意变量相关性大于0.4的变量
##var1: other vars, var2: significant_vars
for var1 in data.columns:
    if var1 not in significant_vars and var1 != "district" and var1 != "city" and var1 != "region":
        for var2 in significant_vars:
            corr_value = data.corr(method='spearman').loc[var1, var2]
            if abs(corr_value) > 0.4:
                significant_vars.append(var1)
                break
    
# 打印最终筛选结果
print(significant_vars)

# 将数据集的相关列提取出来
new_data = data[significant_vars]

# 保存为新的CSV文件
new_data.to_csv('../data/Correlation_Data_Sumed.csv', index=False)

######################################################
#########以下是全部的关联性分析结果，用于分析，设置筛选阈值
# 进行关联性分析
corr = data.corr(method = "spearman")

# 保存结果到 result.csv 文件
corr.to_csv('../data/Correlation_Results.csv')
#print(corr)

# 绘制热力图
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
plt.show()
