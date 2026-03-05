import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
import os
import pickle
from collections import Counter


class LiteraryPersonaSAGE:
    def __init__(self, n_personas=8, alpha=1.0, iterations=50):
        self.P = n_personas
        self.alpha = alpha
        self.iters = iterations
        self.eta_pers = None
        self.eta_bg = None
        self.vocab = None
        self.X = None  # Sparse matrix of character-word counts (C x V)
        self.char_info = []  # List of char_key info

    def fit(self, df):
        print("Preparing data...")
        df["char_key"] = df["book"] + "_" + df["char_id"].astype(str)
        df["feat"] = df["role"] + ":" + df["word"].astype(str)

        # Build vocabulary
        self.vocab = sorted(df["feat"].unique())
        v_map = {feat: i for i, feat in enumerate(self.vocab)}
        V = len(self.vocab)
        print(f"Vocab size: {V}")

        # Build characters
        char_keys = sorted(df["char_key"].unique())
        c_map = {ck: i for i, ck in enumerate(char_keys)}
        C = len(char_keys)
        print(f"Characters: {C}")

        # Collect character info for output
        temp_info = (
            df.groupby("char_key")[["author", "book"]].first().reindex(char_keys)
        )
        self.char_info = temp_info.reset_index().to_dict("records")

        # Build sparse matrix
        print("Building sparse matrix...")
        row_indices = df["char_key"].map(c_map).values
        col_indices = df["feat"].map(v_map).values
        data = df["count"].values
        self.X = csr_matrix((data, (row_indices, col_indices)), shape=(C, V))

        # Assignments
        p_assignments = np.random.randint(0, self.P, size=C)

        print("Calculating background distribution...")
        all_feats_counts = np.array(self.X.sum(axis=0)).flatten()
        self.eta_bg = np.log(
            (all_feats_counts + 0.1) / (np.sum(all_feats_counts) + 0.1)
        )

        self.eta_pers = np.zeros((self.P, V))

        print(f"Starting EM training ({self.iters} iterations)...")
        for it in range(self.iters):
            # M-step: update eta_pers
            for p in range(self.P):
                mask = p_assignments == p
                if np.any(mask):
                    counts_p = np.array(self.X[mask].sum(axis=0)).flatten()
                    total_p = np.sum(counts_p)
                    log_pp = np.log((counts_p + 0.1) / (total_p + 0.1))
                    self.eta_pers[p] = log_pp - self.eta_bg
                else:
                    self.eta_pers[p] = np.zeros(V)

            # E-step: update assignments
            persona_counts = Counter(p_assignments)

            # Optimization: Pre-calculate log_probs for each persona
            # This is still memory-intensive if V is huge, but let's try.
            # log_probs = eta_bg + eta_pers - logsumexp(eta_bg + eta_pers)
            all_log_probs = np.zeros((self.P, V))
            for p in range(self.P):
                logits = self.eta_bg + self.eta_pers[p]
                all_log_probs[p] = logits - logsumexp(logits)

            # Now update each character's assignment
            # This part is still O(C * P) but with matrix multiplication we can speed up term2
            # term2 = X * all_log_probs.T (C x V) * (V x P) -> (C x P)
            term2_all = self.X.dot(all_log_probs.T)

            for i in range(C):
                log_posteriors = np.zeros(self.P)
                old_p = p_assignments[i]
                persona_counts[old_p] -= 1

                for p in range(self.P):
                    term1 = np.log(persona_counts[p] + self.alpha)
                    term2 = term2_all[i, p]
                    log_posteriors[p] = term1 + term2

                probs = np.exp(log_posteriors - logsumexp(log_posteriors))
                new_p = np.random.choice(self.P, p=probs)
                p_assignments[i] = new_p
                persona_counts[new_p] += 1

            if (it + 1) % 5 == 0:
                print(f"Iteration {it+1}/{self.iters} completed.")

        self.p_assignments = p_assignments
        self.save_model()

    def save_model(
        self, model_path="sage_model_full.pkl", results_path="sage_results_full.csv"
    ):
        print(f"Saving model to {model_path}...")
        model_data = {
            "eta_bg": self.eta_bg,
            "eta_pers": self.eta_pers,
            "vocab": self.vocab,
            "n_personas": self.P,
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Saving results to {results_path}...")
        df_results = pd.DataFrame(self.char_info)
        df_results["persona"] = self.p_assignments
        df_results.to_csv(results_path, index=False)

        features_data = []
        for p in range(self.P):
            top_idx = np.argsort(self.eta_pers[p])[-50:][::-1]
            for idx in top_idx:
                features_data.append(
                    {
                        "persona": p,
                        "feature": self.vocab[idx],
                        "weight": self.eta_pers[p][idx],
                    }
                )
        pd.DataFrame(features_data).to_csv("persona_features_full.csv", index=False)
        print("Done.")


if __name__ == "__main__":
    csv_path = "./fullset_data/all_words.csv"
    if os.path.exists(csv_path):
        print("Loading data...")
        df = pd.read_csv(csv_path)

        print("Filtering characters...")
        df["temp_key"] = df["book"] + "_" + df["char_id"].astype(str)
        char_totals = df.groupby("temp_key")["count"].sum()
        valid_keys = char_totals[char_totals >= 10].index
        df_filtered = df[df["temp_key"].isin(valid_keys)].copy()

        print(f"Characters with >= 10 mentions: {len(valid_keys)}")

        model = LiteraryPersonaSAGE(n_personas=8, iterations=30)
        model.fit(df_filtered)
    else:
        print(f"Error: {csv_path} not found.")
