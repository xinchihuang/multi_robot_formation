import matplotlib.pyplot as plt

# 商品名称和销量
products = ['A', 'B', 'C', 'D']
sales = [50, 150, 100, 90]

# 创建柱状图
plt.figure(figsize=(10, 6))
plt.bar(products, sales)

# 添加标题和坐标轴标签
plt.title('Sales of Products')
plt.xlabel('Product')
plt.ylabel('Sales')

# 显示图表
plt.show()
