import os
import torch
import pandas as pd
import numpy as np
from collections import Counter
from sage.model import SAGE_CVAE_Flat

def main():
    # Configuration
    checkpoint_root = "checkpoints"
    vocab_file = "data/processed/female_vocab_map.csv"
    output_dir = "data/results/refined_keywords"
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
    M = 1 # Found to be 1 from data analysis
    
    all_summary_records = []

    for ckpt_dir in ckpt_dirs:
        P_str = ckpt_dir.split('_')[1][1:]
        P = int(P_str)
        print(f"\n>>> Processing P={P} from {ckpt_dir}...")
        
        try:
            # Paths
            ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "best_model.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "latest_model.pt")
            
            if not os.path.exists(ckpt_path):
                print(f"    [Warning] No model found in {ckpt_dir}, skipping.")
                continue
                
            # Load model state
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Verify M from state_dict if possible
            if 'decoder.eta_author' in state_dict:
                M_actual = state_dict['decoder.eta_author'].shape[0]
                if M_actual != M:
                    print(f"    [Info] Updating M from {M} to {M_actual} based on checkpoint.")
                    M = M_actual

            # Initialize model with correct P
            model = SAGE_CVAE_Flat(V, M, P, R)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            eta_persona = model.decoder.eta_persona.detach().numpy() # [P, R, V]
            
            refined_records = []
            
            for r_idx in range(R):
                r_name = role_names[r_idx]
                
                # 1. Collect top words for each persona for this role
                top_k_initial = 100 
                persona_top_indices = []
                for p in range(P):
                    weights = eta_persona[p, r_idx, :]
                    # Use indices where weight > 0
                    pos_indices = np.where(weights > 0)[0]
                    if len(pos_indices) == 0:
                        persona_top_indices.append([])
                        continue
                    
                    # Sort by weight descending
                    sorted_pos = pos_indices[np.argsort(weights[pos_indices])[::-1]]
                    persona_top_indices.append(sorted_pos[:top_k_initial].tolist())
                
                # 2. Count occurrences across personas for THIS role
                # Cross-persona deduplication: if a word appears in >1 persona for this role, it's not "distinctive"
                word_counts = Counter()
                for p_indices in persona_top_indices:
                    word_counts.update(p_indices)
                
                # 3. Filter: Keep only words that appear in EXACTLY ONE persona for this role
                for p in range(P):
                    unique_indices = [idx for idx in persona_top_indices[p] if word_counts[idx] == 1]
                    
                    # Take top 30 unique words for the persona-role description
                    for idx in unique_indices[:30]:
                        word = vocab[idx]
                        weight = eta_persona[p, r_idx, idx]
                        refined_records.append({
                            'n_personas': P,
                            'persona': p,
                            'role': r_name,
                            'word': word,
                            'weight': weight
                        })
            
            # Save CSV for this P
            df_refined = pd.DataFrame(refined_records)
            out_path = os.path.join(output_dir, f"refined_keywords_P{P}.csv")
            df_refined.to_csv(out_path, index=False)
            print(f"    [Success] Saved {len(df_refined)} refined keywords to {out_path}")
            
            # Append to all summary
            all_summary_records.extend(refined_records)
            
        except Exception as e:
            print(f"    [Error] Processing {ckpt_dir} failed: {e}")
            import traceback
            traceback.print_exc()

    # Save a master summary file
    if all_summary_records:
        df_master = pd.DataFrame(all_summary_records)
        master_path = os.path.join(output_dir, "refined_keywords_all_P.csv")
        df_master.to_csv(master_path, index=False)
        print(f"\n>>> Global summary saved to {master_path}")

if __name__ == "__main__":
    main()
