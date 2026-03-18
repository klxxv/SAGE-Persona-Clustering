import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sage.model import SAGE_CVAE_Flat, AdvancedLiterarySAGE

def main():
    # Configuration
    checkpoint_root = "checkpoints"
    vocab_file = "data/processed/female_vocab_map.csv"
    data_file = "fullset_data/all_words.csv"
    if not os.path.exists(data_file):
        data_file = "data/processed/all_words.csv"
    
    output_dir = "data/results/char_assignments"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f">>> Loading data from {data_file}...")
    # Use the trainer's logic to ensure consistency in feature building
    trainer = AdvancedLiterarySAGE(n_personas=8) # Dummy P
    df = trainer.load_data(data_file, vocab_file)
    
    # Pre-mapping
    char_keys = sorted(df["char_key"].unique())
    trainer.char_map = {ck: i for i, ck in enumerate(char_keys)}
    trainer.C = len(trainer.char_map)
    df['c_idx'] = df['char_key'].map(trainer.char_map)
    
    # Prepare Features (Normalized BoW)
    print(">>> Preparing character features...")
    char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    char_feats = np.zeros((trainer.C, trainer.V))
    char_feats[char_word_counts['c_idx'], char_word_counts['w_idx']] = char_word_counts['count']
    char_feats_ts = torch.tensor(char_feats, dtype=torch.float32)
    char_feats_ts = F.normalize(char_feats_ts, p=2, dim=1)
    
    # Find checkpoint directories
    ckpt_dirs = [d for d in os.listdir(checkpoint_root) if d.startswith("cvae_P") and "_L1e-06" in d]
    ckpt_dirs.sort(key=lambda x: int(x.split('_')[1][1:]))
    
    V, R = trainer.V, 4
    M = 1 

    for ckpt_dir in ckpt_dirs:
        P_str = ckpt_dir.split('_')[1][1:]
        P = int(P_str)
        print(f"\n>>> Exporting assignments for P={P} from {ckpt_dir}...")
        
        try:
            ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "best_model.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(checkpoint_root, ckpt_dir, "latest_model.pt")
            
            if not os.path.exists(ckpt_path):
                continue
                
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            if 'decoder.eta_author' in state_dict:
                M = state_dict['decoder.eta_author'].shape[0]

            model = SAGE_CVAE_Flat(V, M, P, R)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Predict Persona Probabilities
            with torch.no_grad():
                logits = model.encoder(char_feats_ts)
                probs = F.softmax(logits, dim=-1).numpy()
            
            # Save results
            records = []
            for i, char_key in enumerate(char_keys):
                prob_vec = ",".join([f"{p:.6f}" for p in probs[i]])
                records.append({
                    'Character id': char_key,
                    'Persona Prob Vec': prob_vec
                })
            
            df_out = pd.DataFrame(records)
            out_path = os.path.join(output_dir, f"char_assignments_P{P}.csv")
            df_out.to_csv(out_path, index=False)
            print(f"    [Success] Saved assignments for {len(df_out)} characters.")
            
        except Exception as e:
            print(f"    [Error] Failed for {ckpt_dir}: {e}")

if __name__ == "__main__":
    main()
