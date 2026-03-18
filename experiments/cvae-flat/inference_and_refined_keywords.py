import os
import torch
import pandas as pd
import numpy as np
from sage.model import SAGE_CVAE_Flat

def main():
    # Configuration
    checkpoint_root = "checkpoints"
    vocab_file = "data/processed/female_vocab_map.csv"
    output_dir = "data/results/all_persona_keywords"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vocab
    print(f">>> Loading vocabulary from {vocab_file}...")
    df_vocab = pd.read_csv(vocab_file)
    vocab = df_vocab['word'].tolist()
    V = len(vocab)
    
    # Find checkpoint directories
    ckpt_dirs = [d for d in os.listdir(checkpoint_root) if d.startswith("cvae_P") and "_L1e-06" in d]
    ckpt_dirs.sort(key=lambda x: int(x.split('_')[1][1:])) # Sort by P
    
    role_names = {0: 'agent', 1: 'patient', 2: 'possessive', 3: 'predicative'}
    R = 4
    M = 1 
    
    all_summary_records = []

    for ckpt_dir in ckpt_dirs:
        P_str = ckpt_dir.split('_')[1][1:]
        P = int(P_str)
        print(f"\n>>> Extracting all keywords for P={P} from {ckpt_dir}...")
        
        try:
            ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "best_model.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "latest_model.pt")
            
            if not os.path.exists(ckpt_path):
                print(f"    [Warning] No model found in {ckpt_dir}, skipping.")
                continue
                
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            if 'decoder.eta_author' in state_dict:
                M = state_dict['decoder.eta_author'].shape[0]

            model = SAGE_CVAE_Flat(V, M, P, R)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            eta_persona = model.decoder.eta_persona.detach().numpy() # [P, R, V]
            
            records = []
            top_k = 50 # 提取每个 Persona 的前 50 个词
            
            for p in range(P):
                for r_idx in range(R):
                    r_name = role_names[r_idx]
                    weights = eta_persona[p, r_idx, :]
                    
                    # 获取权重最高的前 K 个词的索引
                    top_indices = np.argsort(weights)[::-1][:top_k]
                    
                    for rank, idx in enumerate(top_indices):
                        weight = weights[idx]
                        if weight <= 0: continue # 仅保留正向相关的词
                        
                        records.append({
                            'n_personas': P,
                            'persona': p,
                            'role': r_name,
                            'rank': rank + 1,
                            'word': vocab[idx],
                            'weight': float(weight)
                        })
            
            # 保存该参数下的所有词汇
            df_p = pd.DataFrame(records)
            out_path = os.path.join(output_dir, f"all_keywords_P{P}.csv")
            df_p.to_csv(out_path, index=False)
            print(f"    [Success] Exported {len(df_p)} keywords to {out_path}")
            
            all_summary_records.extend(records)
            
        except Exception as e:
            print(f"    [Error] Failed for {ckpt_dir}: {e}")

    # 保存总表
    if all_summary_records:
        df_all = pd.DataFrame(all_summary_records)
        all_path = os.path.join(output_dir, "all_keywords_combined.csv")
        df_all.to_csv(all_path, index=False)
        print(f"\n>>> Combined summary saved to {all_path}")

if __name__ == "__main__":
    main()
