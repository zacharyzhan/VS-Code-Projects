""" import matplotlib.pyplot as plt

# 绘制数据
plt.plot([1, 2, 3, 4], [5, 6, 7, 8])
plt.show()

# 结束文件
print("This is the end of the test file")

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # 选择中间元素作为基准
    left = [x for x in arr if x < pivot]  # 小于基准的元素
    middle = [x for x in arr if x == pivot]  # 等于基准的元素
    right = [x for x in arr if x > pivot]  # 大于基准的元素
    return quick_sort(left) + middle + quick_sort(right)  # 递归排序

# 测试快速排序
data = [3, 6, 8, 10, 1, 2, 1]
sorted_data = quick_sort(data)
print("排序后的数组:", sorted_data)


# 该脚本用于测试文件的头部输出和列表元素的打印
print("This is the head of the test file")
list = ['red', 'green', 'blue', 'yellow', 'white', 'black']
# 打印列表中的第一个、第二个和第三个元素
print( list[0] )
print( list[1] )
print( list[2] )

# 打印列表中的最后一个、倒数第二个和倒数第三个元素
print( list[-1] )
print( list[-2] )
print( list[-3] ) """

print("This is the end of the test file")