import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sage.model import SAGE_CVAE_Flat

def get_cvae_probs(model, char_feats_ts):
    with torch.no_grad():
        logits = model.encoder(char_feats_ts)
        return F.softmax(logits, dim=-1).numpy()

def main():
    checkpoint_root = "checkpoints"
    vocab_file = "data/processed/female_vocab_map.csv"
    target_file = "data/processed/target_female_ids.csv"
    output_dir = "data/results/char_assignments"
    os.makedirs(output_dir, exist_ok=True)
    
    # Target characters
    df_targets = pd.read_csv(target_file)
    target_ids = df_targets['character_id'].tolist()
    
    # Load vocab
    df_vocab = pd.read_csv(vocab_file)
    vocab = df_vocab['word'].tolist()
    V = len(vocab)
    word_map = {w: i for i, w in enumerate(vocab)}

    # Features for the 153 targets from female_words_base.csv
    df_base = pd.read_csv("data/processed/female_words_base.csv")
    df_base = df_base[df_base['character_id'].isin(target_ids)]
    df_base = df_base[df_base['word'].isin(word_map)]
    df_base['w_idx'] = df_base['word'].map(word_map)
    
    id_to_row = {tid: i for i, tid in enumerate(target_ids)}
    char_feats = np.zeros((len(target_ids), V))
    for _, row in df_base.iterrows():
        rid = id_to_row[row['character_id']]
        char_feats[rid, int(row['w_idx'])] = row['freq']
    
    char_feats_ts = torch.tensor(char_feats, dtype=torch.float32)
    char_feats_ts = F.normalize(char_feats_ts, p=2, dim=1)

    ckpt_dirs = [d for d in os.listdir(checkpoint_root) if d.startswith("cvae_P") or d in ["P4", "P8", "P12", "P16"]]

    for d in ckpt_dirs:
        print(f"\n>>> Exporting Persona Probability Vector for model: {d}")
        
        try:
            is_cvae = d.startswith("cvae_P")
            if is_cvae:
                ckpt_path = os.path.join(checkpoint_root, d, "best_model.pt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(checkpoint_root, d, "latest_model.pt")
            else:
                ckpt_path = os.path.join(checkpoint_root, d, "checkpoint_it1000.pt")
            
            if not os.path.exists(ckpt_path): continue
            
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            if is_cvae:
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                # M=24 based on earlier info
                M = state_dict['decoder.eta_author'].shape[0] if 'decoder.eta_author' in state_dict else 24
                P = state_dict['decoder.eta_persona'].shape[0]
                model = SAGE_CVAE_Flat(V, M, P, 4)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                
                probs = get_cvae_probs(model, char_feats_ts)
            else:
                # Traditional models (P4, P8, P12, P16)
                P = checkpoint['P']
                p_assign = checkpoint['p_assignments']
                
                # Derive probabilities from character assignments (One-hot as proxy)
                probs = np.zeros((len(target_ids), P))
                for i, tid in enumerate(target_ids):
                    if tid < len(p_assign):
                        assigned_p = int(p_assign[tid])
                        probs[i, assigned_p] = 1.0
            
            records = []
            for i, tid in enumerate(target_ids):
                prob_vec = ",".join([f"{v:.6f}" for v in probs[i]])
                records.append({'Character id': tid, 'Persona Prob Vec': prob_vec})
            
            df_out = pd.DataFrame(records)
            out_path = os.path.join(output_dir, f"char_assignments_{d}.csv")
            df_out.to_csv(out_path, index=False)
            print(f"    [Success] Exported assignments for {len(df_out)} target characters.")
            
        except Exception as e:
            print(f"    [Error] Failed for {d}: {e}")

if __name__ == "__main__":
    main()
