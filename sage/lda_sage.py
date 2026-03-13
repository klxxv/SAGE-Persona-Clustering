import torch
import torch.nn as nn
import torch.nn.functional as F

class LDASAGE(nn.Module):
    """
    Advanced LDA-SAGE with Author Bias, Dependency Roles, and Dirichlet Prior
    """
    def __init__(self, vocab_size, n_chars, n_authors, n_personas=8, n_roles=4):
        super(LDASAGE, self).__init__()
        self.V = vocab_size
        self.C = n_chars
        self.M = n_authors
        self.P = n_personas
        self.R = n_roles
        
        # 1. Character-Persona Distribution (LDA Logits)
        # Initialize with enough variance to allow the Dirichlet prior to push it
        self.theta_logits = nn.Parameter(torch.randn(n_chars, n_personas) * 1.0)
        
        # 2. SAGE Biases
        self.eta_p = nn.Parameter(torch.randn(n_personas, vocab_size) * 0.01) 
        self.eta_m = nn.Parameter(torch.randn(n_authors, vocab_size) * 0.01) 
        self.tau_r = nn.Parameter(torch.randn(n_roles, vocab_size) * 0.01)    
        
        # Background Log Frequencies
        self.register_buffer('bg_log_freq', torch.zeros(vocab_size))

    def get_persona_dist(self, char_idx=None):
        if char_idx is not None:
            return F.softmax(self.theta_logits[char_idx], dim=-1)
        return F.softmax(self.theta_logits, dim=-1)

    def forward(self, char_idx, author_idx, role_idx):
        # 1. log P(p|c) - (batch, P)
        log_theta = F.log_softmax(self.theta_logits[char_idx], dim=-1)
        
        # 2. log P(w|p, m, r)
        bg = self.bg_log_freq.view(1, 1, -1)
        eta_p = self.eta_p.unsqueeze(0) 
        eta_m = self.eta_m[author_idx].unsqueeze(1) 
        tau_r = self.tau_r[role_idx].unsqueeze(1)   
        
        logits_p = bg + eta_p + eta_m + tau_r
        log_prob_w_given_p = F.log_softmax(logits_p, dim=-1) 
        
        # 3. Marginalize: log sum_p exp( log P(p|c) + log P(w|p, m, r) )
        combined_log_prob = log_theta.unsqueeze(-1) + log_prob_w_given_p
        
        # Return marginalized log-probs AND the log_theta (for Dirichlet penalty)
        return torch.logsumexp(combined_log_prob, dim=1), log_theta

def lda_sage_loss(log_probs, counts, log_theta_batch, alpha=0.1, l1_lambda_eta=1e-4, model=None):
    """
    Loss with Dirichlet Prior Penalty:
    Loss = -NLL - log P(theta | alpha)
    where log P(theta | alpha) = sum (alpha - 1) * log theta
    """
    total_tokens = counts.sum() + 1e-10
    
    # 1. Negative Log-Likelihood (Data fitting)
    neg_ll = -torch.sum(log_probs * counts) / total_tokens
    
    # 2. Dirichlet Penalty (Persona Sparsity)
    # If alpha < 1, (alpha - 1) is negative. 
    # Penalty = - (alpha - 1) * sum(log_theta) -> + (1 - alpha) * sum(log_theta)
    # Since log_theta is always negative, this term is negative. 
    # Minimizing it pushes one log_theta to 0 and others to -inf.
    dirichlet_penalty = - (alpha - 1.0) * torch.mean(torch.sum(log_theta_batch, dim=1))
    
    # 3. SAGE Sparsity (Word Biases)
    reg_loss = 0
    if model is not None:
        reg_loss += l1_lambda_eta * (torch.sum(torch.abs(model.eta_p)) + torch.sum(torch.abs(model.eta_m))) / (model.V * (model.P + model.M))
    
    return neg_ll + dirichlet_penalty + reg_loss
