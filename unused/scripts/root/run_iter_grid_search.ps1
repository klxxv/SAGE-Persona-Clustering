cd sage
# 请确保在运行此脚本前已执行: conda activate semantics
python -u train_serial.py `
    --word_csv_file ../data/processed/word2vec_clusters.csv `
    --data_file ../data/processed/all_words.csv `
    --output_dir ../checkpoints `
    --output_json ../data/results/grid_search_results_iters.json `
    --n_personas_list 4 8 12 16 `
    --em_iters_list 200 400 600 800 1000 `
    --l1_lambda 1e-6
