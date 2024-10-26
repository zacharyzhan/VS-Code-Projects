import os
import csv

def predict_score(test_file, result_dir):
  """
  预测测试文件的分数。

  Args:
    test_file: 测试文件的路径。
    result_dir: 包含已打分结果文件的目录路径。

  Returns:
    一个字典，包含每个结果文件的预测分数。
  """

  predicted_scores = {}  # 存储每个结果文件的预测分数

  with open(test_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    test_data = list(reader)

  for filename in os.listdir(result_dir):
    if filename.endswith('.csv'):
      filepath = os.path.join(result_dir, filename)
      score = float(filename[:-4])  # 从文件名中提取分数

      with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        result_data = list(reader)

      same_count = 0
      for i in range(len(test_data)):
        if test_data[i][1] == result_data[i][1]:  # 比较预测结果
          same_count += 1

      predicted_scores[filename] = score * same_count / len(test_data)

  return predicted_scores

# 示例用法
test_file = r"C:\Users\zacha\Downloads\1-87.csv"
result_dir = 'results'
predicted_scores = predict_score(test_file, result_dir)

for filename, score in predicted_scores.items():
  print(f'{filename}: {score}')
