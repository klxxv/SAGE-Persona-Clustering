import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralSAGE(nn.Module):
    """
    变分神经 SAGE (Variational Neural SAGE)
    核心：使用摊销推理 (Amortized Inference) 替代传统的 Gibbs 采样。
    """
    def __init__(self, vocab_size, n_personas=8, n_roles=4, hidden_dim=256, dropout=0.2):
        super(NeuralSAGE, self).__init__()
        self.V = vocab_size
        self.P = n_personas
        self.R = n_roles
        
        # 1. Encoder (Inference Network)
        # 输入：角色的 Bag-of-Words (BoW)
        # 输出：Persona 分布的均值和方差 (Logistic Normal)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, n_personas)
        self.fc_logvar = nn.Linear(hidden_dim // 2, n_personas)
        
        # 2. Decoder (Generative Network - SAGE Additive Structure)
        # 遵循 SAGE 论文公式: P(w|theta, r) \propto exp(m_w + eta_{p,w} + tau_{r,w})
        # eta_p: Persona 特征偏向 (P x V)
        # tau_r: Role 特征偏向 (R x V)
        # m_w: 背景词频 (固定)
        self.eta_p = nn.Parameter(torch.randn(n_personas, vocab_size) * 0.02)
        self.tau_r = nn.Parameter(torch.randn(n_roles, vocab_size) * 0.02)
        self.register_buffer('bg_log_freq', torch.zeros(vocab_size)) 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return F.softmax(z, dim=-1) # 使用 Softmax 确保归一化为角色权重

    def forward(self, bow_vec, role_idx):
        """
        bow_vec: (batch, V)
        role_idx: (batch,)
        """
        # Encoder -> Latent Persona Distribution
        h = self.encoder(bow_vec)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Sample Persona weights theta
        theta = self.reparameterize(mu, logvar) # (batch, P)
        
        # Decoder -> SAGE Additive Probability
        # log_prob \propto bg + (theta @ eta) + tau[role]
        batch_size = bow_vec.size(0)
        
        # (batch, P) @ (P, V) -> (batch, V)
        persona_effect = torch.matmul(theta, self.eta_p) 
        # (batch, V)
        role_effect = self.tau_r[role_idx]
        
        # Total log-odds
        logits = self.bg_log_freq + persona_effect + role_effect
        log_reconstruction = F.log_softmax(logits, dim=-1)
        
        return log_reconstruction, mu, logvar, theta

def vae_loss(recon_x, x, mu, logvar, theta, eta_p, l1_lambda=1e-4, entropy_lambda=1e-3):
    """
    强化稀疏性的 ELBO 损失函数
    1. Reconstruction Loss: 负对数似然
    2. KL Divergence: 变分约束
    3. Entropy Penalty: 强制 theta 集中 (类似 Dirichlet alpha < 1)
    4. L1 Regularization: 强制 eta_p 稀疏 (SAGE 核心)
    """
    # 1. 重构损失
    recon_loss = -(recon_x * x).sum(dim=-1).mean()
    
    # 2. KL 散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    # 3. 熵惩罚 (Entropy Penalty on theta)
    # 熵越小，theta 越趋向于 one-hot (即一个角色只属于一个类型)
    entropy = -torch.sum(theta * torch.log(theta + 1e-10), dim=-1).mean()
    
    # 4. L1 稀疏惩罚 (on SAGE parameters eta_p)
    l1_loss = torch.norm(eta_p, 1)
    
    total_loss = recon_loss + kl_loss + (entropy_lambda * entropy) + (l1_lambda * l1_loss)
    
    return total_loss, recon_loss, kl_loss, entropy, l1_loss
