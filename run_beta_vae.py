import os
import time
import pandas as pd
import numpy as np
import torch
from sage.beta_vae_model import BetaLiterarySAGE

def check_cuda_environment():
    print("="*50)
    print(">>> CUDA Environment Check")
    print("="*50)
    if torch.cuda.is_available():
        print(f"CUDA is available: YES")
        print(f"Device Count: {torch.cuda.device_count()}")
        return True
    else:
        print("CUDA is available: NO")
        return False

def run_beta_vae():
    has_gpu = check_cuda_environment()
    
    word_csv = "fullset_data/word2vec_clusters.csv" 
    data_file = "fullset_data/all_words.csv"
    
    if not os.path.exists(word_csv):
        word_csv = "data/processed/word2vec_clusters.csv"
        data_file = "data/processed/all_words.csv"

    n_axes = 5
    beta = 2.0  # beta > 1 鼓励解耦
    iters = 500 if has_gpu else 50
    
    print(f">>> Initializing Beta-VAE with n_axes={n_axes}, beta={beta}, iters={iters}")
    trainer = BetaLiterarySAGE(n_axes=n_axes, beta=beta, l1_lambda=1e-6, iters=iters)
    
    print(">>> Loading dataset...")
    df_full = trainer.load_data(data_file, word_csv)
    
    print(f"\n>>> Starting Training ({iters} iters)...")
    start_time = time.time()
    trainer.fit(df_full)
    end_time = time.time()
    print(f">>> Training completed in {end_time - start_time:.2f}s")
    
    # 提取特征轴的两极词汇
    print("\n>>> Extracting Opposite Poles for Continuous Axes (Disentanglement)...")
    vocab = trainer.vocab
    eta_axes = trainer.model.decoder.eta_axes.detach().cpu().numpy() # [K, R, V]
    
    role_names = {0: 'Agent', 1: 'Patient', 2: 'Possessive', 3: 'Predicative'}
    
    for k in range(n_axes):
        print(f"\n" + "="*40)
        print(f"  Axis {k} Spectrum")
        print("="*40)
        
        for r_idx, r_name in role_names.items():
            if r_idx < trainer.R:
                weights = eta_axes[k, r_idx, :]
                
                # 正向极点 (Positive Pole)
                pos_indices = np.argsort(weights)[-5:][::-1]
                pos_words = [vocab[i] for i in pos_indices if weights[i] > 0.1]
                
                # 负向极点 (Negative Pole)
                neg_indices = np.argsort(weights)[:5]
                neg_words = [vocab[i] for i in neg_indices if weights[i] < -0.1]
                
                print(f"  [{r_name}]")
                print(f"    (+) High End: {', '.join(pos_words) if pos_words else '(None)'}")
                print(f"    (-) Low End : {', '.join(neg_words) if neg_words else '(None)'}")

if __name__ == "__main__":
    run_beta_vae()
