import torch
import torch.nn as nn
import torch.nn.functional as F

class LDASAGE(nn.Module):
    """
    LDA-SAGE 模型 (混合角色类型模型)
    
    1. LDA 部分: 每个角色是一个 Persona 的混合分布 (theta_c)
    2. SAGE 部分: 每个 Persona (z) 对词汇的影响是加性的 log-linear 偏置
    """
    def __init__(self, vocab_size, n_chars, n_personas=8, n_roles=4):
        super(LDASAGE, self).__init__()
        self.V = vocab_size
        self.C = n_chars
        self.P = n_personas
        self.R = n_roles
        
        # 1. 角色-Persona 分布 (LDA)
        # 存储每个角色的 Persona 分数 (未归一化)
        self.theta_logits = nn.Parameter(torch.randn(n_chars, n_personas) * 0.1)
        
        # 2. SAGE 偏置 (Generative Model)
        self.eta_p = nn.Parameter(torch.randn(n_personas, vocab_size) * 0.01)
        self.tau_r = nn.Parameter(torch.randn(n_roles, vocab_size) * 0.01)
        
        # 背景词频 (固定的, 作为 Baseline)
        self.register_buffer('bg_log_freq', torch.zeros(vocab_size))

    def get_persona_dist(self):
        """返回每个角色的 Persona 混合分布 (归一化)"""
        return F.softmax(self.theta_logits, dim=-1)

    def forward(self, char_idx, role_idx):
        """
        计算给定角色和角色类型下，所有词汇的生成概率。
        P(w|c, r) = sum_p [ P(p|c) * P(w|p, r) ]
        """
        # 1. P(p|c) - (batch, P)
        theta = F.softmax(self.theta_logits[char_idx], dim=-1)
        
        # 2. 计算每个 Persona 下所有词的 log-prob - (P, V)
        # 这里为了计算 P(w|p, r) 需要考虑所有的 P
        # log_prob_p \propto bg + eta_p + tau_r
        # 注意：这里的 tau_r 是对应 batch 里的每个实例的，需要广播
        
        # (batch, P, V)
        persona_effect = self.eta_p.unsqueeze(0).expand(char_idx.size(0), -1, -1)
        role_effect = self.tau_r[role_idx].unsqueeze(1).expand(-1, self.P, -1)
        
        # logits_p: (batch, P, V)
        logits_p = self.bg_log_freq.view(1, 1, -1) + persona_effect + role_effect
        
        # Prob(w|p, r): (batch, P, V)
        prob_w_given_p = F.softmax(logits_p, dim=-1)
        
        # 3. 边际化 Persona: sum_p P(p|c) * P(w|p, r)
        # (batch, 1, P) @ (batch, P, V) -> (batch, 1, V)
        prob_w = torch.bmm(theta.unsqueeze(1), prob_w_given_p).squeeze(1)
        
        return torch.log(prob_w + 1e-10) # 返回 log-prob 用于计算 Likelihood

def lda_sage_loss(log_probs, counts, theta_logits, l1_lambda=1e-5):
    """
    损失函数 = -对数似然 + 稀疏正则化
    """
    # 负对数似然 (加权词频)
    neg_ll = -torch.sum(log_probs * counts)
    
    # 稀疏正则化 (SAGE 的核心)
    # 我们希望 eta_p 是稀疏的 (每个 Persona 只有少量关键词)
    # 也希望 theta 是相对集中的 (一个角色不要混太多种类)
    sparsity_loss = l1_lambda * torch.sum(torch.abs(theta_logits))
    
    return neg_ll + sparsity_loss
