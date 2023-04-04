import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import f_oneway

# Read data
data = pd.read_csv('../data/2.1_Train_Data_Sumed_For_Process.csv')

# Preprocess the data
# Apply one-hot encoding to categorical features
categorical_features = ['date', 'district', 'zip code', 'city']

# Save the list of categorical variables with p-value less than 0.05
new_categorical_features = []

# Using ANOVA to calculate the correlation
# For each categorical variable, calculate the ANOVA F-value and p-value with the numerical target variable
for feature in categorical_features:
    group = data.groupby(feature)["total cost"].apply(list)
    f, p = f_oneway(*group)
    if p < 0.05 and f > 45:
        new_categorical_features.append(feature)
        print(f"Feature {feature}, F-value: {f:.2f}, p-value: {p:.4f}")
        
# Preserve strongly correlated categorical features
new_data = data[new_categorical_features]




# Get the column names of all non-categorical features
numerical_features = [col for col in data.columns if col not in categorical_features + ["total cost"]]

# Keep only numeric variables and target variables
numerical_data = data[numerical_features + ["total cost"]]

corr_matrix = numerical_data.corr(method='pearson', numeric_only = True)

# Draw a heat map...
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.show()

# Save the correlation coefficient matrix as a CSV file
corr_matrix.to_csv('../data/2.2_Correlation_Matrix.csv')

# Calculate the correlation between each variable and "total cost"
corr = data.corr(method = "spearman", numeric_only = True)['total cost']

# Filter variables with an absolute value of correlation greater than or equal to 0.35
significant_vars = corr[(abs(corr) >= 0.35) & (corr.index != 'total cost')].index.tolist()
print(significant_vars)

# Find variables other than significant_vars that have a correlation greater than 0.4 with any variable in significant_vars
##var1: other vars, var2: significant_vars
for var1 in data.columns:
    if var1 not in significant_vars and var1 != "district" and var1 != "city" and var1 != "zip code" and var1 != "date":
        for var2 in significant_vars:
            corr_value = data.corr(method='spearman', numeric_only = True).loc[var1, var2]
            if abs(corr_value) > 0.4:
                significant_vars.append(var1)
                break
    
# Print the significant values
print(significant_vars)

new_numerical_data = data[significant_vars]
# Using concat() to splice two data frames together by columns, where axis=1 means splicing by columns
new_data = pd.concat([new_data, new_numerical_data], axis=1)

# Save the result
new_data.to_csv('../data/2.3_Correlation_Data_Sumed.csv', index=False)



# Read the data
data = pd.read_csv('../data/2.3_Correlation_Data_Sumed.csv')

# Define the classifier of total cost, to cost rank
def classify_total_cost(total_cost):
    if total_cost >= 0 and total_cost < 300000:
        return 1
    elif total_cost >= 300000 and total_cost < 500000:
        return 2
    elif total_cost >= 500000 and total_cost < 700000:
        return 3
    else:
        return 4

# Apply a categorical function and add a new column
data['cost rank'] = data['total cost'].apply(lambda x: classify_total_cost(x))
# delete the total cost variable
data.drop(['total cost'], axis=1, inplace=True)

# Save
data.to_csv('../data/2.4_Train_Data_New.csv', index=False)
selected_columns = data.columns.tolist()
selected_columns.remove('cost rank')
print(selected_columns)

# Read the test data and save it in the new style.
test_data = pd.read_csv('../data/Test_Data.csv')
new_test_data = test_data[selected_columns]
new_test_data.loc[:, 'cost rank'] = ''
new_test_data.to_csv('../data/2.4_Test_Data_New.csv', index=False)