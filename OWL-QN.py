import torch
from torch.optim import Optimizer

class OWLQN(Optimizer):
    """
    改进版 Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) 近似实现
    结合了伪梯度(Pseudo-gradient)、正交象限投影(Orthant Projection) 以及简单的回溯线搜索(Backtracking Line Search)
    """
    def __init__(self, params, lr=1.0, l1_lambda=1.0, c1=1e-4, beta=0.5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, l1_lambda=l1_lambda, c1=c1, beta=beta)
        super(OWLQN, self).__init__(params, defaults)

    def _pseudo_gradient(self, w, g, l1_lambda):
        """
        计算次梯度/伪梯度 (Pseudo-gradient)
        处理 L1 正则化在 0 点不可导的问题
        """
        if l1_lambda == 0.0:
            return g # 对于没有正则化的参数组（如 eta_bg），直接返回原始梯度

        pg = torch.zeros_like(g)
        
        # w < 0 的象限
        idx_neg = w < 0
        pg[idx_neg] = g[idx_neg] - l1_lambda
        
        # w > 0 的象限
        idx_pos = w > 0
        pg[idx_pos] = g[idx_pos] + l1_lambda
        
        # w == 0 的象限
        idx_zero = w == 0
        g_zero = g[idx_zero]
        pg_zero = torch.zeros_like(g_zero)
        
        # 只有当原始梯度大于 L1 惩罚时，才会在 0 点产生向左或向右的梯度
        pg_zero[g_zero + l1_lambda < 0] = g_zero[g_zero + l1_lambda < 0] + l1_lambda
        pg_zero[g_zero - l1_lambda > 0] = g_zero[g_zero - l1_lambda > 0] - l1_lambda
        
        pg[idx_zero] = pg_zero
        return pg

    def _project_to_orthant(self, x, x_old, pg):
        """
        将更新后的权重投影回原本的正交象限
        防止跨越坐标轴改变符号，如果符号改变则强制截断为 0
        """
        # 定义目标象限：由当前变量所在的象限或伪梯度的反方向决定
        orthant = torch.sign(x_old)
        orthant[orthant == 0] = torch.sign(-pg[orthant == 0])
        
        # 将跨越象限的值截断为 0
        x_projected = x.clone()
        cross_mask = (torch.sign(x_projected) * orthant) < 0
        x_projected[cross_mask] = 0.0
        
        return x_projected

    @torch.no_grad()
    def step(self, closure):
        """
        执行单步 OWL-QN 更新，包含闭包求值和线搜索
        """
        if closure is None:
            raise RuntimeError("OWL-QN requires a closure to evaluate loss for line search.")
            
        # 1. 计算初始 Loss 和梯度
        with torch.enable_grad():
            loss = closure()
            
        initial_loss = loss.item()

        # 保存当前参数和计算出的伪梯度
        p_olds = []
        pgs = []
        
        for group in self.param_groups:
            l1_lambda = group['l1_lambda']
            for p in group['params']:
                if p.grad is None:
                    p_olds.append(None)
                    pgs.append(None)
                    continue
                    
                p_olds.append(p.clone())
                pg = self._pseudo_gradient(p, p.grad, l1_lambda)
                pgs.append(pg)

        # 2. 回溯线搜索 (Backtracking Line Search) 确保充分下降
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta'] # 步长衰减率
            l1_lambda = group['l1_lambda']
            
            idx = 0
            for p in group['params']:
                if p.grad is None:
                    idx += 1
                    continue
                
                p_old = p_olds[idx]
                pg = pgs[idx]
                
                current_lr = lr
                max_ls_iters = 10 # 最大线搜索次数
                
                for ls_iter in range(max_ls_iters):
                    # 尝试走一步
                    p.copy_(p_old)
                    p.add_(pg, alpha=-current_lr)
                    
                    # 正交投影
                    p.copy_(self._project_to_orthant(p, p_old, pg))
                    
                    # 计算尝试步之后的 loss
                    with torch.enable_grad():
                        new_loss = closure()
                        
                    # L1 正则化的目标值计算
                    l1_penalty_old = l1_lambda * p_old.abs().sum()
                    l1_penalty_new = l1_lambda * p.abs().sum()
                    
                    # 简单的充分下降条件检查 (Armijo rule 简化版)
                    if new_loss.item() + l1_penalty_new <= initial_loss + l1_penalty_old:
                        break # 找到了合适的步长
                        
                    # 否则衰减学习率
                    current_lr *= beta
                    
                idx += 1

        return loss