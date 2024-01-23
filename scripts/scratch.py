import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('/plots/ViT_4/1.csv', skiprows=3)
df=df.iloc[:, 2:]
# 计算每列的非空元素数量
non_null_counts = df.count()
print(non_null_counts[:20])
# 确定要选取的列数
N = 54  # 比如，我们想选取非空元素数量最多的前 3 列

# 找出非空元素数量最多的 N 列的名称
top_columns = non_null_counts.nlargest(N).index
print(top_columns)
# 保持原始列顺序，选取这些列
selected_columns = [col for col in df.columns if col in top_columns]

# 获取选中的列
selected_df = df[selected_columns]

# 查看结果
print(selected_df.columns)