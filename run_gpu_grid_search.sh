#!/bin/bash

# 1. 启动脚本：run_gpu_grid_search.sh
# 建议在 GPU 服务器上使用单进程或是少量并发（取决于显存）
# 如果显存足够（>24G），可以保留 --n_personas_list 同时跑多个

echo ">>> Starting SAGE GPU Grid Search..."

# 确保输出目录存在
mkdir -p checkpoints
mkdir -p data/results

# 使用 nohup 后台运行，并重定向输出
cd sage
nohup python3 -u train.py \
    --word_csv_file ../data/processed/word2vec_clusters.csv \
    --data_file ../data/processed/all_words.csv \
    --output_dir ../checkpoints \
    --output_json ../data/results/grid_search_results_full.json \
    --n_personas_list 4 8 12 16 \
    --em_iters_list 50 100 150 200 > ../data/results/gpu_run.log 2>&1 &

echo ">>> Task started in background. Use ./monitor_gpu.sh to check progress."
