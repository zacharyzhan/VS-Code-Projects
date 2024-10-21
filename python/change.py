import pandas as pd

# 读取CSV文件
df = pd.read_csv(r"C:\Users\zacha\Desktop\cla_pre_vgg_mixed.csv")

# 假设第二列的名称是 'column_2'，将 0 和 1 互换
df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: 1 if x == 0 else 0)

# 保存修改后的CSV文件
df.to_csv('output_file.csv', index=False, header=True, encoding='utf-8')

print("output_file.csv")
