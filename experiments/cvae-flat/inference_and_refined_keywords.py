import os
import torch
import pandas as pd
import numpy as np
from sage.model import SAGE_CVAE_Flat

def main():
    checkpoint_root = "checkpoints"
    vocab_file = "data/processed/female_vocab_map.csv"
    output_dir = "data/results/all_persona_keywords"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vocab
    df_vocab = pd.read_csv(vocab_file)
    vocab = df_vocab['word'].tolist()
    V = len(vocab)
    
    # Identify all checkpoint directories (CVAE and Traditional)
    ckpt_dirs = [d for d in os.listdir(checkpoint_root) if d.startswith("cvae_P") or d in ["P4", "P8", "P12", "P16"]]
    
    role_names = {0: 'agent', 1: 'patient', 2: 'possessive', 3: 'predicative'}
    R = 4
    
    all_summary_records = []

    for ckpt_dir in ckpt_dirs:
        print(f"\n>>> Extracting ALL keywords for model: {ckpt_dir}")
        
        try:
            # 1. Determine Model Type and Load Weights
            is_cvae = ckpt_dir.startswith("cvae_P")
            if is_cvae:
                ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "best_model.pt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "latest_model.pt")
            else:
                ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "checkpoint_it1000.pt")
            
            if not os.path.exists(ckpt_path): continue
            
            # 2. Extract Persona Weights (eta_persona)
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            if is_cvae:
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                eta_persona = state_dict['decoder.eta_persona'].detach().numpy() # [P, R, V]
                P = eta_persona.shape[0]
            else:
                # Traditional models (LiteraryPersonaSAGE)
                model_weights = checkpoint['model_weights']
                # The weights are in model_weights['eta_pers'] which is [P, R, V]
                eta_persona = model_weights['eta_pers'].detach().numpy()
                P = eta_persona.shape[0]

            records = []
            top_k = 50 
            
            for p in range(P):
                for r_idx in range(R):
                    r_name = role_names[r_idx]
                    weights = eta_persona[p, r_idx, :]
                    
                    # Sort by weight descending
                    top_indices = np.argsort(weights)[::-1][:top_k]
                    
                    for rank, idx in enumerate(top_indices):
                        weight = float(weights[idx])
                        if weight <= 0: continue 
                        
                        records.append({
                            'model': ckpt_dir,
                            'persona': p,
                            'role': r_name,
                            'rank': rank + 1,
                            'word': vocab[idx],
                            'weight': weight
                        })
            
            # Save CSV for this model
            df_m = pd.DataFrame(records)
            out_path = os.path.join(output_dir, f"keywords_{ckpt_dir}.csv")
            df_m.to_csv(out_path, index=False)
            print(f"    [Success] Exported {len(df_m)} keywords for {ckpt_dir}")
            
            all_summary_records.extend(records)
            
        except Exception as e:
            print(f"    [Error] Failed for {ckpt_dir}: {e}")

    # Master summary
    if all_summary_records:
        pd.DataFrame(all_summary_records).to_csv(os.path.join(output_dir, "all_keywords_combined.csv"), index=False)
        print(f"\n>>> Combined summary saved to data/results/all_persona_keywords/all_keywords_combined.csv")

if __name__ == "__main__":
    main()
