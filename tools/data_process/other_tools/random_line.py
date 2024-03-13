import random

# 读取文件中的所有行
with open('/mnt/public/usr/wangmingze/Datasets/CD/WHU-EAST_ASIA/test.txt', 'r') as f:
    lines = f.readlines()

# 计算需要抽取的行数
lines_to_extract = len(lines) // 4

# 随机抽取行
selected_lines = random.sample(lines, lines_to_extract)

# 将选中的行写入新文件
with open('/mnt/public/usr/wangmingze/Datasets/CD/WHU-EAST_ASIA/test_0.25.txt', 'w') as f:
    f.writelines(selected_lines)