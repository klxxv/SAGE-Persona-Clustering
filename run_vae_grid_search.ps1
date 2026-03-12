# VAE-SAGE 网格搜索脚本
# 搜索参数: n_personas 和 epochs

$personas_list = @(4, 8, 12, 16)
$epochs_list = @(50, 100, 200)

$word_csv = "../data/processed/word2vec_clusters.csv"
$data_file = "../data/processed/all_words.csv"
$base_output = "../data/results/vae_grid_search"

# 确保在运行此脚本前已执行: conda activate semantics
cd sage/neural

foreach ($p in $personas_list) {
    foreach ($e in $epochs_list) {
        $run_name = "P${p}_E${e}"
        $output_dir = "${base_output}/${run_name}"
        
        Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
        Write-Host ">>> Starting VAE Grid Search: Personas=$p, Epochs=$e" -ForegroundColor Green
        Write-Host ">>> Output: $output_dir" -ForegroundColor Gray
        Write-Host ("=" * 60) + "`n"

        # 1. 训练
        python -u train_vae.py `
            --word_csv_file $word_csv `
            --data_file $data_file `
            --output_dir $output_dir `
            --n_personas $p `
            --epochs $e `
            --save_interval 50

        # 2. 评估
        python -u eval_vae.py `
            --word_csv_file $word_csv `
            --data_file $data_file `
            --model_dir $output_dir `
            --n_personas $p
            
        Write-Host ">>> Finished $run_name" -ForegroundColor Green
    }
}

Write-Host "`n>>> VAE Grid Search Complete!" -ForegroundColor Cyan
