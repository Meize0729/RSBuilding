import random

def split_file_randomly(input_file_path, train_file_path, test_file_path, split_ratio=0.8):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    random.seed(42)
    # 随机打乱行
    random.shuffle(lines)

    # 计算分割点
    split_point = int(len(lines) * split_ratio)

    # 将数据分为训练集和测试集
    train_lines = lines[:split_point]
    test_lines = lines[split_point:]

    # 写入train.txt
    with open(train_file_path, 'w') as train_file:
        train_file.writelines(train_lines)

    # 写入test.txt
    with open(test_file_path, 'w') as test_file:
        test_file.writelines(test_lines)

# 示例：调用函数
split_file_randomly('/mnt/public/usr/wangmingze/Datasets/CD/SECOND/train.txt', '/mnt/public/usr/wangmingze/Datasets/CD/SECOND/train_split.txt', '/mnt/public/usr/wangmingze/Datasets/CD/SECOND/test_split.txt')