import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Hierarchical Softmax SAGE Model
# ==========================================
class HierarchicalSAGE(nn.Module):
    def __init__(self, M, P, num_internal_nodes):
        super().__init__()
        self.eta_bg = nn.Parameter(torch.zeros(num_internal_nodes))
        self.eta_meta = nn.Parameter(torch.zeros(M, num_internal_nodes))
        self.eta_pers = nn.Parameter(torch.zeros(P, num_internal_nodes))
        
    def forward(self, m_idx, p_idx, node_paths, node_signs):
        """
        Calculates log probabilities for a batch of words.
        m_idx, p_idx: (batch_size,)
        node_paths: (batch_size, max_path_len)
        node_signs: (batch_size, max_path_len)
        """
        batch_size = node_paths.shape[0]
        max_path_len = node_paths.shape[1]

        # Expand m_idx and p_idx for indexing
        m_idx_expanded = m_idx.unsqueeze(1).expand(-1, max_path_len)
        p_idx_expanded = p_idx.unsqueeze(1).expand(-1, max_path_len)

        # Advanced indexing to get weights for nodes in paths
        bg = self.eta_bg[node_paths]
        meta = self.eta_meta[m_idx_expanded, node_paths]
        pers = self.eta_pers[p_idx_expanded, node_paths]
        
        logits = bg + meta + pers
        
        # Use a mask for valid paths (non-padding nodes)
        path_mask = (node_paths != -1).float()
        
        log_probs = F.logsigmoid(node_signs * logits) * path_mask
        word_log_probs = log_probs.sum(dim=1)
        
        return word_log_probs

# ==========================================
# 2. Main SAGE Trainer Class
# ==========================================
class LiteraryPersonaSAGE:
    def __init__(self, n_personas=8, alpha=1.0, l1_lambda=0.1, iterations=50, min_mentions=10):
        self.P = n_personas
        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.iters = iterations
        self.min_mentions = min_mentions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _build_balanced_tree(self, vocab):
        print(">>> Building a balanced binary tree for Hierarchical Softmax...")
        vocab_map = {name: i for i, name in enumerate(vocab)}
        
        # Recursive function to build the tree
        internal_nodes = []
        def build_recursive(current_vocab_indices):
            if len(current_vocab_indices) == 1:
                return vocab_map[current_vocab_indices[0]] # Return leaf index
            
            # Create a new internal node
            node_id = len(internal_nodes)
            internal_nodes.append({'children': []})
            
            # Split vocab and recurse
            mid = len(current_vocab_indices) // 2
            left_child = build_recursive(current_vocab_indices[:mid])
            right_child = build_recursive(current_vocab_indices[mid:])
            
            internal_nodes[node_id]['children'] = [left_child, right_child]
            return node_id + len(vocab) # Internal nodes are indexed after leaves

        # Build the tree structure
        root_node_idx = build_recursive(vocab)

        # Generate paths and signs for each word
        paths = {}
        def generate_paths_recursive(node_idx, current_path, current_signs):
            if node_idx < len(vocab): # It's a leaf node
                paths[vocab[node_idx]] = (current_path, current_signs)
                return

            node = internal_nodes[node_idx - len(vocab)]
            # Left child
            generate_paths_recursive(node['children'][0], current_path + [node_idx - len(vocab)], current_signs + [-1.0])
            # Right child
            generate_paths_recursive(node['children'][1], current_path + [node_idx - len(vocab)], current_signs + [1.0])
        
        generate_paths_recursive(root_node_idx, [], [])
        
        num_internal_nodes = len(internal_nodes)
        max_len = max(len(p[0]) for p in paths.values())
        
        # Pad paths and create tensors
        self.word_paths = -torch.ones((len(vocab), max_len), dtype=torch.long)
        self.word_signs = torch.zeros((len(vocab), max_len))

        for i, word in enumerate(vocab):
            path, signs = paths[word]
            self.word_paths[i, :len(path)] = torch.tensor(path)
            self.word_signs[i, :len(signs)] = torch.tensor(signs)
            
        print(f"Tree built. Vocab size: {len(vocab)}, Internal nodes: {num_internal_nodes}")
        return num_internal_nodes

    def load_and_preprocess_data(self, data_file, cluster_file):
        # This function remains the same
        df = pd.read_csv(data_file)
        df_clusters = pd.read_csv(cluster_file)
        word_to_cluster = dict(zip(df_clusters.word, df_clusters.cluster_id))
        df['cluster_id'] = df['word'].map(word_to_cluster)
        df.dropna(subset=['cluster_id'], inplace=True)
        df['cluster_id'] = df['cluster_id'].astype(int)
        df["char_key"] = df["book"] + "_" + df["char_id"].astype(str)
        char_totals = df.groupby("char_key")["count"].sum()
        valid_keys = char_totals[char_totals >= self.min_mentions].index
        df = df[df["char_key"].isin(valid_keys)].copy()
        df["feat"] = df["role"] + ":" + df["cluster_id"].astype(str)
        return df

    def fit(self, df):
        # --- 1. Build Vocab, Tree, and Data Matrices ---
        self.vocab = sorted(df["feat"].unique())
        self.v_map = {feat: i for i, feat in enumerate(self.vocab)}
        self.V = len(self.vocab)
        
        num_internal_nodes = self._build_balanced_tree(self.vocab)
        self.word_paths = self.word_paths.to(self.device)
        self.word_signs = self.word_signs.to(self.device)

        authors = sorted(df["author"].unique())
        self.m_map = {author: i for i, author in enumerate(authors)}
        self.M = len(authors)

        char_keys = sorted(df["char_key"].unique())
        c_map = {ck: i for i, ck in enumerate(char_keys)}
        self.C = len(char_keys)
        print(f"Clustered Vocab (V): {self.V}, Authors (M): {self.M}, Chars (C): {self.C}")

        temp_info = df.groupby("char_key")[["author", "book"]].first().reindex(char_keys)
        self.char_info_df = temp_info.reset_index()
        char_to_m = np.array([self.m_map[a] for a in self.char_info_df['author']])
        char_to_book = self.char_info_df['book'].values
        unique_books = self.char_info_df['book'].unique()

        row_indices = df["char_key"].map(c_map).values
        col_indices = df["feat"].map(self.v_map).values
        data = df["count"].values
        self.X = sp.csr_matrix((data, (row_indices, col_indices)), shape=(self.C, self.V))

        np.random.seed(42)
        self.p_assignments = np.random.randint(0, self.P, size=self.C)
        self.book_persona_counts = defaultdict(lambda: np.zeros(self.P, dtype=int))
        for c_idx in range(self.C):
            self.book_persona_counts[char_to_book[c_idx]][self.p_assignments[c_idx]] += 1

        # --- 2. Initialize Model and Optimizer ---
        self.model = HierarchicalSAGE(self.M, self.P, num_internal_nodes).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)

        print(f">>> Starting Stochastic EM training ({self.iters} rounds)...")
        # --- 3. EM Training Loop ---
        for it in range(self.iters):
            print(f"\n--- Round {it+1}/{self.iters} ---")
            
            # --- M-STEP ---
            self.model.train()
            u_indices = char_to_m * self.P + self.p_assignments
            U_total = self.M * self.P
            G = sp.csr_matrix((np.ones(self.C), (u_indices, np.arange(self.C))), shape=(U_total, self.C))
            Y = G.dot(self.X)
            
            active_u_pairs, word_indices = Y.nonzero()
            counts = Y.data
            
            total_loss = 0
            total_tokens = 0
            
            # Process in mini-batches for memory efficiency
            batch_size = 1024
            for i in tqdm(range(0, len(active_u_pairs), batch_size), desc="  [M-Step] Optimizing"):
                optimizer.zero_grad()
                
                batch_indices = active_u_pairs[i:i+batch_size]
                batch_words = word_indices[i:i+batch_size]
                batch_counts = torch.tensor(counts[i:i+batch_size], dtype=torch.float32, device=self.device)

                m_idx = torch.tensor(batch_indices // self.P, device=self.device)
                p_idx = torch.tensor(batch_indices % self.P, device=self.device)
                
                # Get paths for the words in the batch
                node_paths = self.word_paths[batch_words]
                node_signs = self.word_signs[batch_words]

                word_log_probs = self.model(m_idx, p_idx, node_paths, node_signs)
                
                batch_total_tokens = batch_counts.sum()
                if batch_total_tokens > 0:
                    nll_loss = -torch.sum(word_log_probs * batch_counts) / batch_total_tokens
                    nll_loss.backward()
                    optimizer.step()
                    total_loss += nll_loss.item() * batch_total_tokens
                    total_tokens += batch_total_tokens

            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            print(f"  [M-Step] Avg NLL/Token: {avg_loss:.4f}")

            # --- E-STEP ---
            self.model.eval()
            with torch.no_grad():
                all_word_log_probs = torch.zeros((self.M, self.P, self.V), device=self.device)
                # Pre-calculate log probs for all words for all author-persona pairs
                for m_idx in range(self.M):
                    for p_idx in range(self.P):
                        m_tensor = torch.full((self.V,), m_idx, device=self.device, dtype=torch.long)
                        p_tensor = torch.full((self.V,), p_idx, device=self.device, dtype=torch.long)
                        all_word_log_probs[m_idx, p_idx, :] = self.model(m_tensor, p_tensor, self.word_paths, self.word_signs)

                for book in tqdm(unique_books, desc="  [E-Step] Gibbs Sampling"):
                    char_indices_in_book = np.where(char_to_book == book)[0]
                    if len(char_indices_in_book) == 0: continue
                    m_d = char_to_m[char_indices_in_book[0]]
                    
                    log_probs_for_author = all_word_log_probs[m_d, :, :] # Shape (P, V)

                    for c in char_indices_in_book:
                        old_p = self.p_assignments[c]
                        self.book_persona_counts[book][old_p] -= 1
                        
                        prior = np.log(self.book_persona_counts[book] + self.alpha)
                        
                        c_word_vector = torch.tensor(self.X[c].toarray().flatten(), dtype=torch.float32, device=self.device)
                        ll = (log_probs_for_author * c_word_vector).sum(dim=1)
                        
                        post_logits = torch.tensor(prior, device=self.device) + ll
                        post_probs = F.softmax(post_logits, dim=0).cpu().numpy()
                        
                        new_p = np.random.choice(self.P, p=post_probs)
                        self.p_assignments[c] = new_p
                        self.book_persona_counts[book][new_p] += 1
                        
    def save_results(self, output_dir):
        # This method is modified to save the hierarchical model correctly
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n>>> Saving model and results to {output_dir}/")

        torch.save(self.model.state_dict(), os.path.join(output_dir, "sage_model_weights.pt"))
        
        metadata = {
            "vocab": self.vocab, "m_map": self.m_map, "P": self.P,
            # Save tree structure for post-processing
            "word_paths": self.word_paths.cpu(),
            "word_signs": self.word_signs.cpu()
        }
        with open(os.path.join(output_dir, "sage_metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        df_results = self.char_info_df.copy()
        df_results["persona"] = self.p_assignments
        df_results.to_csv(os.path.join(output_dir, "sage_character_personas.csv"), index=False)
        print(">>> All files saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hierarchical SAGE model.")
    parser.add_argument('--cluster_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_file', type=str, default='fullset_data/all_words.csv')
    parser.add_argument('--n_personas', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=10) # Reduced for feasibility
    parser.add_argument('--l1_lambda', type=float, default=0.0) # L1 is not used in this version
    
    args = parser.parse_args()

    if os.path.exists(args.cluster_file):
        model = LiteraryPersonaSAGE(n_personas=args.n_personas, iterations=args.iterations)
        df_processed = model.load_and_preprocess_data(args.data_file, args.cluster_file)
        model.fit(df_processed)
        model.save_results(args.output_dir)
    else:
        print(f"Error: Cluster file not found at {args.cluster_file}")

