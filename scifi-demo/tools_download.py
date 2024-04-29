import os
import sys
import requests
import glob

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

url = "https://huggingface.co/datasets/zxbsmk/webnovel_cn/resolve/main/novel_cn_token512_50k.json?download=true"
save_path = "data/scifi-finetune.json"
download_file(url, save_path)

sys.exit(0)

# 小说模型训练数据集下载链接: https://pan.baidu.com/s/1bC8fH8hyt28L9pV3fjOHIQ 提取码: 9i9g
def find_txt_files(directory):
    return glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

def concatenate_txt_files(files, output_file):
    with open(output_file, 'w') as outfile:
        for file in files:
            with open(file, 'r') as infile:
                outfile.write(infile.read() + '\n')  # Adds a newline between files

directory = 'data'
output_file = 'data/scifi.txt'

# Find all .txt files
txt_files = find_txt_files(directory)

# Concatenate all found .txt files into one
concatenate_txt_files(txt_files, output_file)
