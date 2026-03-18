import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from sage.model import AdvancedLiterarySAGE
from sage.metrics import calculate_silhouette, calculate_latent_silhouette, calculate_flat_perplexity

def check_cuda_environment():
    print("="*50)
    print(">>> CUDA Environment Check")
    print("="*50)
    if torch.cuda.is_available():
        print(f"CUDA is available: YES")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        print("="*50 + "\n")
        return True
    else:
        print("CUDA is available: NO")
        print("Warning: Training on CPU with full data will be extremely slow.")
        print("="*50 + "\n")
        return False

def run_full_cvae(n_personas=8, iters=1000, batch_size=8192, subset_chars=None, lr=1e-3, l1_lambda=1e-6, resume_path=None, data_file=None, word_csv=None):
    # 1. Check Environment
    has_gpu = check_cuda_environment()
    
    # 2. Setup paths with fallback and validation
    if data_file is None or word_csv is None:
        # Default candidates
        candidates = [
            ("fullset_data/all_words.csv", "fullset_data/word2vec_clusters.csv"),
            ("data/processed/all_words.csv", "data/processed/word2vec_clusters.csv"),
            ("data/processed/female_words_with_ids.csv", "data/processed/female_vocab_map.csv")
        ]
        
        found = False
        for d, w in candidates:
            if os.path.exists(d) and os.path.exists(w):
                data_file, word_csv = d, w
                found = True
                break
        
        if not found:
            print("Error: Required data files not found in default locations:")
            for d, w in candidates:
                print(f"  - Tried: {d} AND {w}")
            print("\nPlease specify paths manually using --data_file and --word_csv")
            return
    else:
        if not os.path.exists(data_file) or not os.path.exists(word_csv):
            print(f"Error: Manually specified files not found:")
            print(f"  - data_file: {data_file} ({'Exists' if os.path.exists(data_file) else 'NOT FOUND'})")
            print(f"  - word_csv: {word_csv} ({'Exists' if os.path.exists(word_csv) else 'NOT FOUND'})")
            return

    print(f">>> Using Data: {data_file}")
    print(f">>> Using Vocab: {word_csv}")

    # 3. Initialize Trainer
    print(f">>> Initializing CVAE-SAGE: P={n_personas}, iters={iters}, batch_size={batch_size}, lr={lr}, l1={l1_lambda}")
    trainer = AdvancedLiterarySAGE(n_personas=n_personas, mode='cvae_flat', iters=iters, l1_lambda=l1_lambda)
    
    # 4. Load & Global Mapping
    print(">>> Loading dataset and performing global ID mapping...")
    df_full = trainer.load_data(data_file, word_csv)
    
    if subset_chars is not None:
        unique_chars = df_full['char_key'].unique()
        subset_keys = unique_chars[:subset_chars]
        df_full = df_full[df_full['char_key'].isin(subset_keys)].copy()
        print(f"    Subsetting to first {subset_chars} characters.")

    authors = sorted(df_full["author"].unique())
    trainer.m_map = {a: i for i, a in enumerate(authors)}
    trainer.M = len(trainer.m_map)
    
    char_keys = sorted(df_full["char_key"].unique())
    trainer.char_map = {ck: i for i, ck in enumerate(char_keys)}
    trainer.C = len(trainer.char_map)
    
    print(f"    Global Total: {trainer.M} authors, {trainer.C} characters, {trainer.V} tokens.")
    
    # 5. Train/Test Split
    np.random.seed(42)
    np.random.shuffle(char_keys)
    
    split_idx = int(len(char_keys) * 0.9)
    train_keys = char_keys[:split_idx]
    test_keys = char_keys[split_idx:]
    
    train_df = df_full[df_full['char_key'].isin(train_keys)].copy()
    test_df = df_full[df_full['char_key'].isin(test_keys)].copy()
    
    print(f"    Train: {len(train_keys)} characters | Test: {len(test_keys)} characters")

    # 6. Fit model
    checkpoint_dir = f'checkpoints/cvae_P{n_personas}_L{l1_lambda}'
    print(f"\n>>> Starting Training (Checkpoints: {checkpoint_dir})...")
    start_time = time.time()
    train_df_mapped = trainer.fit(train_df, batch_size=batch_size, checkpoint_dir=checkpoint_dir, lr=lr, resume_path=resume_path)
    end_time = time.time()
    print(f">>> Training completed in {end_time - start_time:.2f}s")
    
    # Save final model
    os.makedirs('data/results', exist_ok=True)
    model_name = f'cvae_flat_P{n_personas}_iters{iters}_l1{l1_lambda}.pt'
    model_path = os.path.join('data/results', model_name)
    torch.save(trainer.model.state_dict(), model_path)
    print(f">>> Final model saved to {model_path}")

    # 7. Evaluate
    print("\n>>> Evaluating Metrics...")
    latent_s_score, labels = calculate_latent_silhouette(trainer.model, train_df_mapped, trainer.device)
    
    test_df_mapped = trainer.prepare_df(test_df)
    if len(test_df_mapped) > 0:
        perp = calculate_flat_perplexity(trainer.model, test_df_mapped, trainer.device)
    else:
        perp = float('nan')
    
    print(f"    [Latent Space] Silhouette Score: {latent_s_score:.4f}")
    print(f"    [Test Set] Perplexity          : {perp:.4f}")

    # 8. Feature Keywords
    print("\n>>> Extracting Top Words for learned Personas...")
    vocab = trainer.vocab
    eta_persona = trainer.model.decoder.eta_persona.detach().cpu().numpy()
    role_names = {0: 'Agent', 1: 'Patient', 2: 'Possessive', 3: 'Predicative'}
    
    records = []
    unique_personas = np.unique(labels)
    for p in range(n_personas):
        assigned_count = (labels == p).sum() if p in unique_personas else 0
        print(f"\n  Persona {p} (Assigned to {assigned_count} characters):")
        for r_idx, r_name in role_names.items():
            if r_idx < trainer.R:
                weights = eta_persona[p, r_idx, :]
                top_indices = np.argsort(weights)[-10:][::-1]
                top_words = [vocab[i] for i in top_indices if weights[i] > 0]
                for word in top_words:
                    records.append({'persona': p, 'role': r_name, 'word': word, 'assigned_chars': assigned_count})
                if top_words:
                    print(f"    {r_name}: {', '.join(top_words)}")

    keywords_path = f'data/results/keywords_P{n_personas}_l1{l1_lambda}.csv'
    pd.DataFrame(records).to_csv(keywords_path, index=False)
    print(f"\n>>> Keywords saved to {keywords_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full CVAE-SAGE for Production")
    parser.add_argument("--n_personas", type=int, default=8, help="Number of personas")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l1", type=float, default=1e-6, help="L1 penalty lambda")
    parser.add_argument("--subset", type=int, default=None, help="Subset of characters (for testing)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data_file", type=str, default=None, help="Path to main all_words.csv")
    parser.add_argument("--word_csv", type=str, default=None, help="Path to word2vec_clusters.csv")
    
    args = parser.parse_args()
    run_full_cvae(n_personas=args.n_personas, iters=args.iters, batch_size=args.batch_size, 
                  subset_chars=args.subset, lr=args.lr, l1_lambda=args.l1, resume_path=args.resume,
                  data_file=args.data_file, word_csv=args.word_csv)
