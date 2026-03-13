
import torch
import os

def debug():
    print("--- .gitignore Content ---")
    if os.path.exists('.gitignore'):
        try:
            with open('.gitignore', 'r', encoding='utf-16') as f:
                print(f.read())
        except:
            with open('.gitignore', 'r', encoding='utf-8', errors='ignore') as f:
                print(f.read())
    
    ckpt_path = 'checkpoints/P4/checkpoint_it1000.pt'
    if os.path.exists(ckpt_path):
        print(f"\n--- Checking Checkpoint: {ckpt_path} ---")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        weights = ckpt['model_weights']
        for k in ['eta_pers', 'eta_bg', 'eta_meta']:
            if k in weights:
                print(f"{k} shape: {weights[k].shape}")
        
        # Check silhouette if available in result.json
        res_path = 'checkpoints/P4/result_it1000.json'
        if os.path.exists(res_path):
            import json
            with open(res_path, 'r') as f:
                res = json.load(f)
                print(f"Result JSON Silhouette: {res.get('silhouette')}")
    else:
        print(f"\nCheckpoint {ckpt_path} not found!")

if __name__ == "__main__":
    debug()
