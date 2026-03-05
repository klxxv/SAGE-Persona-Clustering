import pickle
import numpy as np
import pandas as pd
from scipy.special import logsumexp

class PersonaInference:
    def __init__(self, model_path='sage_model_full.pkl'):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.eta_bg = data['eta_bg']
        self.eta_pers = data['eta_pers']
        self.vocab = data['vocab']
        self.P = data['n_personas']
        self.v_map = {feat: i for i, feat in enumerate(self.vocab)}

    def infer_character(self, character_features):
        V = len(self.vocab)
        word_counts = np.zeros(V)
        for feat, count in character_features.items():
            if feat in self.v_map:
                word_counts[self.v_map[feat]] = count
        
        log_posteriors = np.zeros(self.P)
        for p in range(self.P):
            logits = self.eta_bg + self.eta_pers[p]
            log_probs = logits - logsumexp(logits)
            log_posteriors[p] = np.sum(word_counts * log_probs)
            
        probs = np.exp(log_posteriors - logsumexp(log_posteriors))
        return probs

if __name__ == "__main__":
    # Example test
    model_file = 'sage_model_full.pkl'
    import os
    if os.path.exists(model_file):
        inferer = PersonaInference(model_file)
        test_char = {"agent:said": 5, "agent:thought": 2}
        probs = inferer.infer_character(test_char)
        print("Persona Probabilities:", probs)
        print("Best Persona:", np.argmax(probs))
    else:
        print(f"Model {model_file} not found.")
