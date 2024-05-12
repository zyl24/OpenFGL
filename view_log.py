import os
import pickle



# 定义搜索的根目录
path = "/home/ai2/work/dataset/distrib/fedsubgraph_louvain_1_Cora_client_10/debug/"

# 要搜索的字符串
search_str = "fedavg"

# 初始化一个列表来存储找到的文件路径
found_files = []

# 使用 os.walk() 遍历目录
for dirpath, dirnames, filenames in os.walk(path):
    for filename in filenames:
        # 检查文件名中是否包含指定的字符串
        if search_str in filename:
            # 构造完整的文件路径
            full_path = os.path.join(dirpath, filename)
            # 将路径添加到列表中
            found_files.append(full_path)

# 输出找到的所有文件的路径
for file_name in found_files:
    print(file_name)

    path_file_name = os.path.join(path, filename)
    with open(path_file_name, 'rb') as file:
        log_file = pickle.load(file)
        
    print(log_file["time"])
    print(log_file["metric"][-1])

        