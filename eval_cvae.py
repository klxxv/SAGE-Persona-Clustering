import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sage.model import AdvancedLiterarySAGE, SAGE_CVAE_Flat


def eval_one_cvae(n_personas, l1_lambda, data_file, word_csv, ckpt_base, res_base):
    ckpt_path = os.path.join(ckpt_base, f"cvae_P{n_personas}_L{l1_lambda}", "best_model.pt")
    res_dir = os.path.join(res_base, f"P{n_personas}_L{l1_lambda}")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return

    print(f"\n{'='*60}")
    print(f">>> Evaluating CVAE  P={n_personas}  L1={l1_lambda}")
    print(f"{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load data (sets up vocab, role_map, log_bg, role_mask) ────────────
    trainer = AdvancedLiterarySAGE(n_personas=n_personas, mode='cvae_flat')
    df_full = trainer.load_data(data_file, word_csv)

    # Set maps that are normally created inside fit()
    authors = sorted(df_full["author"].unique())
    trainer.m_map = {a: i for i, a in enumerate(authors)}
    trainer.M = len(trainer.m_map)
    char_keys = sorted(df_full["char_key"].unique())
    trainer.char_map = {ck: i for i, ck in enumerate(char_keys)}
    trainer.C = len(trainer.char_map)

    df_mapped = trainer.prepare_df(df_full)
    df_mapped = df_mapped.dropna(subset=['c_idx', 'w_idx', 'm_idx'])
    df_mapped['c_idx'] = df_mapped['c_idx'].astype(int)
    df_mapped['w_idx'] = df_mapped['w_idx'].astype(int)

    # ── 2. Build model and load checkpoint ───────────────────────────────────
    model = SAGE_CVAE_Flat(
        trainer.V, trainer.M, n_personas, trainer.R,
        role_mask=trainer.role_mask, log_bg=trainer.log_bg
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    # ── 3. Compute character feature matrix and infer personas ───────────────
    cw = df_mapped.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    char_feats = np.zeros((trainer.C, trainer.V), dtype=np.float32)
    char_feats[cw['c_idx'].values, cw['w_idx'].values] = cw['count'].values

    with torch.no_grad():
        cf_tensor = F.normalize(torch.tensor(char_feats, device=device), p=2, dim=1)
        p_assignments = torch.argmax(model.encoder(cf_tensor), dim=-1).cpu().numpy()

    # ── 4. Reconstruct character metadata ────────────────────────────────────
    # female_words_with_ids.csv uses 'character_id' column (not 'char_id')
    id_col = 'character_id' if 'character_id' in df_mapped.columns else 'char_id'
    char_info = (
        df_mapped.groupby("c_idx")[["author", "book", id_col]]
        .first()
        .reindex(range(trainer.C))
    )
    char_info["persona"] = p_assignments

    # Merge with character_id_name.csv to get human-readable name
    df_names = pd.read_csv("data/processed/character_id_name.csv")
    for col in ['book', 'name']:
        df_names[col] = df_names[col].astype(str).str.strip('"')
    df_names['character_id'] = pd.to_numeric(
        df_names['character_id'].astype(str).str.strip('"'), errors='coerce'
    ).astype('Int64')
    char_info[id_col] = pd.to_numeric(char_info[id_col], errors='coerce').astype('Int64')

    df_assign = char_info.reset_index().merge(
        df_names,
        left_on=['book', id_col],
        right_on=['book', 'character_id'],
        how='left'
    )

    os.makedirs(res_dir, exist_ok=True)
    assign_path = os.path.join(res_dir, "char_assignments.csv")
    df_assign[['book', 'name', 'author', 'character_id', 'persona']].to_csv(
        assign_path, index=False
    )
    print(f"  char_assignments saved  ({len(df_assign)} chars) -> {assign_path}")

    # ── 5. Extract raw keywords (top-15 per persona × role, positive only) ───
    vocab = trainer.vocab
    eta_pers = model.decoder.eta_persona.detach().cpu().numpy()
    role_names = {0: 'Agent', 1: 'Patient', 2: 'Possessive', 3: 'Predicative'}

    kw_records, top_words, global_hits = [], [], {}
    for p in range(n_personas):
        for r_idx, r_name in role_names.items():
            if r_idx >= trainer.R:
                continue
            weights = eta_pers[p, r_idx]
            for rank, i in enumerate(np.argsort(weights)[::-1][:20]):
                if weights[i] <= 0:
                    continue
                word = vocab[i]
                key = (r_name, word)
                global_hits[key] = global_hits.get(key, 0) + 1
                top_words.append((p, r_name, word, float(weights[i])))
                if rank < 15:
                    kw_records.append({
                        'persona': p, 'role': r_name,
                        'word': word, 'weight': float(weights[i])
                    })

    kw_path = os.path.join(res_dir, "keywords.csv")
    pd.DataFrame(kw_records).to_csv(kw_path, index=False)
    print(f"  keywords saved          ({len(kw_records)} rows)  -> {kw_path}")

    # ── 6. Filtered keywords: remove words in >50% of personas per role ───────
    threshold = max(1, n_personas * 0.5)
    filtered = [
        {'persona': p, 'role': r, 'word': w, 'weight': wt}
        for p, r, w, wt in top_words
        if global_hits[(r, w)] <= threshold
    ]
    fk_path = os.path.join(res_dir, "filtered-keywords.csv")
    pd.DataFrame(filtered).to_csv(fk_path, index=False)
    print(f"  filtered-keywords saved ({len(filtered)} rows)  -> {fk_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CVAE models: regenerate char_assignments, keywords, filtered-keywords"
    )
    parser.add_argument("--start_p",  type=int,   default=5)
    parser.add_argument("--end_p",    type=int,   default=10)
    parser.add_argument("--l1",       type=float, default=1.0)
    parser.add_argument("--data",     type=str,   default="data/processed/female_words_with_ids.csv")
    parser.add_argument("--word_csv", type=str,   default="data/processed/female_vocab_map.csv")
    parser.add_argument("--ckpt_base",type=str,   default="checkpoints")
    parser.add_argument("--res_base", type=str,   default="data/results/cvae_results")
    args = parser.parse_args()

    for p in range(args.start_p, args.end_p + 1):
        eval_one_cvae(p, args.l1, args.data, args.word_csv, args.ckpt_base, args.res_base)

    print("\n" + "#" * 60)
    print(">>> All CVAE evaluations complete.")
    print("#" * 60)


if __name__ == "__main__":
    main()
