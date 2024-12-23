import pandas as pd

# 读取两个CSV文件
df2 = pd.read_csv(r"C:\Users\zacha\Desktop\cla_pre_ssa.csv")
# df2 = pd.read_csv('output_file.csv')

df1 = pd.read_csv(r"C:\Users\zacha\Desktop\cla_pre.csv")

# 检查df1第一列的值是否在df2第一列中存在，并比较第二列的值
result = []
for index, row in df1.iterrows():
    value = row[0]
    value_in_df2 = df2[df2.iloc[:, 0] == value]
    if not value_in_df2.empty:
        second_value_df1 = row[1]
        second_value_df2 = value_in_df2.iloc[0, 1]
        if second_value_df1 == second_value_df2:
            result.append([value, '存在且第二列值相同'])
        else:
            result.append([value, '存在但第二列值不同'])
    else:
        result.append([value, '不存在'])

# 将结果保存为新的CSV文件
result_df = pd.DataFrame(result, columns=['第一列值', '比较结果'])
result_df.to_csv('comparison_result.csv', index=False,header=False, encoding='utf-8')
