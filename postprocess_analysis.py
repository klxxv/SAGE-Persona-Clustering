import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from itertools import combinations
import torch.nn.functional as F

# A simplified, non-trainable version of the HS model for inference
class HierarchicalSAGE(nn.Module):
    def __init__(self, state_dict):
        num_internal_nodes = state_dict['eta_bg'].shape[0]
        M = state_dict['eta_meta'].shape[0]
        P = state_dict['eta_pers'].shape[0]
        super().__init__()
        self.eta_bg = nn.Parameter(torch.zeros(num_internal_nodes))
        self.eta_meta = nn.Parameter(torch.zeros(M, num_internal_nodes))
        self.eta_pers = nn.Parameter(torch.zeros(P, num_internal_nodes))
        self.load_state_dict(state_dict)

    def forward(self, m_idx, p_idx, node_paths, node_signs):
        batch_size, max_path_len = node_paths.shape
        m_idx_expanded = m_idx.unsqueeze(1).expand(-1, max_path_len)
        p_idx_expanded = p_idx.unsqueeze(1).expand(-1, max_path_len)
        bg = self.eta_bg[node_paths]
        meta = self.eta_meta[m_idx_expanded, node_paths]
        pers = self.eta_pers[p_idx_expanded, node_paths]
        logits = bg + meta + pers
        path_mask = (node_paths != -1).float()
        log_probs = F.logsigmoid(node_signs * logits) * path_mask
        return log_probs.sum(dim=1)

def load_sage_results(input_dir):
    """Loads the SAGE model weights and metadata from a results directory."""
    print(f">>> Loading results from {input_dir}...")
    weights_path = os.path.join(input_dir, "sage_model_weights.pt")
    metadata_path = os.path.join(input_dir, "sage_metadata.pkl")
    if not os.path.exists(weights_path) or not os.path.exists(metadata_path):
        print(f"Error: Could not find model files in {input_dir}.")
        return None, None
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print("Results loaded successfully.")
    return state_dict, metadata

@torch.no_grad()
def get_persona_distributions(state_dict, metadata):
    """Reconstructs the full probability distribution for each persona from the HS model."""
    print(">>> Reconstructing persona distributions from hierarchical model...")
    model = HierarchicalSAGE(state_dict)
    model.eval()
    
    n_personas = metadata['P']
    n_authors = state_dict['eta_meta'].shape[0]
    vocab = metadata['vocab']
    V = len(vocab)

    word_paths = metadata['word_paths']
    word_signs = metadata['word_signs']
    
    # We average over the author effect to get a general persona distribution
    avg_author_effect = model.eta_meta.mean(dim=0)
    model.eta_meta = nn.Parameter(avg_author_effect.unsqueeze(0).expand(n_authors, -1))

    distributions = []
    m_idx = torch.zeros(V, dtype=torch.long) # Use author 0 as representative

    for p in tqdm(range(n_personas), desc="Calculating distributions"):
        p_idx = torch.full((V,), p, dtype=torch.long)
        log_probs = model(m_idx, p_idx, word_paths, word_signs)
        # Convert log probabilities to probabilities
        probs = torch.exp(log_probs).numpy()
        distributions.append(probs)
    
    return np.array(distributions)

def analyze_persona_words(distributions, metadata, top_n=20):
    """Prints the top and bottom N characteristic words for each persona based on probability."""
    vocab = metadata['vocab']
    n_personas = metadata['P']
    
    print("\n--- Persona Characteristic Words (by Probability) ---")
    
    for p in range(n_personas):
        print(f"\n--- Persona {p} ---")
        probs = distributions[p]
        sorted_indices = np.argsort(probs)
        
        top_indices = sorted_indices[-top_n:][::-1]
        bottom_indices = sorted_indices[:top_n]
        
        print(f"  Top {top_n} Words (Most Probable):")
        for i in top_indices:
            print(f"    - {vocab[i]:<20} (Prob: {probs[i]:.6f})")
            
        print(f"\n  Bottom {top_n} Words (Least Probable):")
        for i in bottom_indices:
            print(f"    - {vocab[i]:<20} (Prob: {probs[i]:.6f})")

def analyze_persona_distance(distributions, n_personas):
    """Calculates the Earth Mover's Distance (EMD) between all pairs of persona distributions."""
    print("\n--- Persona Pairwise Distances (Earth Mover's Distance) ---")
    
    distance_matrix = pd.DataFrame(np.zeros((n_personas, n_personas)),
                                   index=[f"P{i}" for i in range(n_personas)],
                                   columns=[f"P{i}" for i in range(n_personas)])
    
    for i, j in combinations(range(n_personas), 2):
        dist = wasserstein_distance(distributions[i], distributions[j])
        distance_matrix.iloc[i, j] = dist
        distance_matrix.iloc[j, i] = dist
        
    print("Distance Matrix:")
    print(distance_matrix.to_string(float_format="%.4f"))
    print("\nNote: Higher values indicate greater difference between personas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process Hierarchical SAGE model results.")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing the saved HS SAGE model results (e.g., ./sage_outputs_hs).')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
    else:
        state_dict, metadata = load_sage_results(args.input_dir)
        
        if state_dict and metadata:
            # Check if it's a hierarchical model
            if 'word_paths' not in metadata:
                print("Error: These results are not from a Hierarchical Softmax model.")
                print("Please run the older post-processing script for this format.")
            else:
                persona_distributions = get_persona_distributions(state_dict, metadata)
                analyze_persona_words(persona_distributions, metadata, top_n=20)
                analyze_persona_distance(persona_distributions, metadata['P'])
