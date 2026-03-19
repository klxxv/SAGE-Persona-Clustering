import os
import time
import pandas as pd
import numpy as np
import torch
import argparse
from sage.model_traditional import LiteraryPersonaSAGE, ROLES


def build_role_cluster_csvs(cluster_dir):
    """Return {role: csv_path} for a per-role cluster directory."""
    return {role: os.path.join(cluster_dir, f"{role}_clusters.csv") for role in ROLES}


def run_traditional_full(n_personas=8, em_iters=100, l1_lambda=1.0,
                         subset_chars=None, data_file=None,
                         cluster_dir=None, label="W2V-Role"):

    if data_file is None:
        data_file = "data/processed/female_words_with_ids.csv"
    if cluster_dir is None:
        cluster_dir = "data/sage_cluster_dataset/w2v-role"

    role_cluster_csvs = build_role_cluster_csvs(cluster_dir)
    for role, path in role_cluster_csvs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cluster CSV for role '{role}': {path}")

    print(f">>> {label} SAGE (per-role): P={n_personas}, iters={em_iters}, l1={l1_lambda}")

    model = LiteraryPersonaSAGE(
        n_personas=n_personas, em_iters=em_iters,
        l1_lambda=l1_lambda, min_mentions=0
    )
    df_processed, num_nodes_list = model.load_and_preprocess_data(data_file, role_cluster_csvs)

    if subset_chars is not None:
        unique_chars = df_processed['char_key'].unique()
        np.random.seed(42)
        subset_keys  = np.random.choice(unique_chars,
                                         size=min(subset_chars, len(unique_chars)),
                                         replace=False)
        df_processed = df_processed[df_processed['char_key'].isin(subset_keys)].copy()

    res_dir  = os.path.abspath(f"data/results/traditional_results/{label}/P{n_personas}_L{l1_lambda}")
    ckpt_dir = os.path.abspath(f"checkpoints/traditional_{label}_P{n_personas}_L{l1_lambda}")
    os.makedirs(res_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_time = time.time()
    model.fit(df_processed, num_nodes_list, checkpoint_dir=ckpt_dir)
    print(f">>> Training completed in {time.time() - start_time:.2f}s")

    # Export character assignments
    df_assign = model.char_info_df.copy()
    df_assign['persona'] = model.p_assignments
    df_assign.to_csv(os.path.join(res_dir, "char_assignments.csv"), index=False)

    # Extract keywords per persona × role
    ROLE_NAMES = ['Agent', 'Patient', 'Possessive', 'Predicative']
    all_word_weights = []
    model.model.eval()
    with torch.no_grad():
        for r, r_name in enumerate(ROLE_NAMES):
            V_r = len(model.vocab_clusters_list[r])
            wp  = model.word_paths_list[r]   # [V_r, L_r]
            ws  = model.word_signs_list[r]   # [V_r, L_r]

            # Persona effects for this role: [P, V_r]
            pers_effects = (
                model.model.eta_pers[r][:, wp] * ws
            ).sum(dim=2)  # [P, V_r]

            bg_effects = (
                model.model.eta_bg[r][wp] * ws
            ).sum(dim=1)  # [V_r]

            leaf_effects = bg_effects.unsqueeze(0) + pers_effects  # [P, V_r]
            leaf_np = leaf_effects.cpu().numpy()

            for p in range(n_personas):
                top_idx = np.argsort(leaf_np[p])[::-1][:15]
                for idx in top_idx:
                    if leaf_np[p, idx] > 0:
                        all_word_weights.append({
                            'persona':    p,
                            'role':       r_name,
                            'cluster_id': model.vocab_clusters_list[r][idx],
                            'weight':     float(leaf_np[p, idx]),
                        })

    pd.DataFrame(all_word_weights).to_csv(
        os.path.join(res_dir, "keywords.csv"), index=False
    )
    print(f">>> Results saved to {res_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_personas",  type=int,   default=8)
    parser.add_argument("--iters",       type=int,   default=100)
    parser.add_argument("--l1",          type=float, default=1.0)
    parser.add_argument("--subset",      type=int,   default=None)
    parser.add_argument("--data_file",   type=str,   default=None)
    parser.add_argument("--cluster_dir", type=str,   default=None)
    parser.add_argument("--label",       type=str,   default="W2V-Role")
    args = parser.parse_args()
    run_traditional_full(
        n_personas=args.n_personas, em_iters=args.iters, l1_lambda=args.l1,
        subset_chars=args.subset, data_file=args.data_file,
        cluster_dir=args.cluster_dir, label=args.label
    )
