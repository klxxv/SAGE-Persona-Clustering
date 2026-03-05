# SAGE 混合效应模型核心实现指南

要正确实现本模型，你需要摒弃简单的“词频相减”逻辑，转向真正的**最优化求解 (Optimization)** 和**层次贝叶斯采样 (Hierarchical Bayesian Sampling)**。

### 1. 数据结构与参数定义

在开始推断之前，必须建立包含以下三个维度的独立参数矩阵，这正是“混合效应”的核心 ：

* 
**全局基础偏差 ($\eta^0$)**：形状为 `(V,)`，表示每个词的基础对数频率 。


* 
**作者效应参数 ($\eta^{meta}$)**：形状为 `(M, V)`，其中 $M$ 是作者总数 。


* 
**角色效应参数 ($\eta^{pers}$)**：形状为 `(P, V)`，其中 $P$ 是预设的 Persona 总数 。



所有参数（除 $\eta^0$ 外）必须初始化为 0，而不是简单的词频对数，并使用均值为 0、尺度为 $\lambda=1$ 的拉普拉斯 (Laplace) 先验进行正则化 。

### 2. 核心概率计算公式 (Log-Linear 得分)

对于给定作者 $m$ 笔下、被分配为角色类型 $p$ 的实体，生成词汇 $v$ 的非归一化对数得分 (Logit) 为 ：


$$Score(v) = \eta^0_v + \eta^{meta}_{m, v} + \eta^{pers}_{p, v}$$

通过 Softmax 函数将得分转化为概率 ：


$$P(v | m, p, \eta) = \frac{\exp(Score(v))}{\sum_{w=1}^V \exp(Score(w))}$$

### 3. 随机 EM 训练流程 (Stochastic EM)

训练过程由两个交替的步骤组成，通常迭代 50 次 ：

#### A. E-步：折叠吉布斯采样 (Collapsed Gibbs Sampling)

**修正点：必须使用小说级别 (Document-level) 的角色统计，而非全局统计。**

遍历每个角色 $c$（属于小说 $d$，作者为 $m$）：

1. 
**计算先验 (Prior)**：统计**当前小说 $d$ 中**除角色 $c$ 之外，分配给各个 Persona 的角色数量分布 。



$$Prior(z) = \log(count(z \in d_{-c}) + \alpha_z)$$


2. 
**计算似然 (Likelihood)**：基于当前 $\eta$ 参数，计算该角色身上所有实际观测词汇在假设类型为 $z$ 时的对数概率总和 。



$$Likelihood(z) = \sum_{v \in c} count(v) \times \log P(v | m, z, \eta)$$


3. 
**重新采样**：计算后验概率 $Posterior(z) = \exp(Prior(z) + Likelihood(z))$ 并归一化，依据此概率为角色 $c$ 抽取一个新的 Persona $p$ 。



#### B. M-步：最优化求解 (Maximization)

**修正点：不能用简单的对数频率，必须跑带有 L1 惩罚的梯度下降或拟牛顿法求极大似然。**

在确定了所有角色的 $p$ 之后，$\eta$ 的更新等价于求解一个多分类逻辑回归问题 。
你需要构建一个目标函数（负对数似然损失 + L1 正则化），并使用优化器将其最小化：

$$Loss = - \sum_{c} \sum_{v \in c} count(v) \times \log P(v | m_c, p_c, \eta) + \lambda \sum |\eta^{meta}| + \lambda \sum |\eta^{pers}|$$

* 
**技术栈建议**：Bamman 原文使用了 OWL-QN 算法来处理 L1 正则化的不可导问题 。如果在纯 Python 环境中手写梯度非常困难，强烈建议在这个 M-步 使用 **PyTorch** 或 **JAX**。你可以将 $\eta$ 定义为 `nn.Parameter`，直接使用自动求导 (Autograd) 和基于 L1 惩罚的优化器（如 AdamW 或简单的 SGD 带 L1 损失）迭代几十步来逼近最优解，这比在 NumPy 中手写逻辑回归的梯度要稳健得多。



### 4. 提取与分析

在完成了所有 EM 迭代后，直接从优化好的参数矩阵 $\eta^{pers}$ 中提取权重。矩阵中为正且数值越大的权重，代表该词与该 Persona 存在越强的正向关联；反之则是负向关联 。由于 M-步中 L1 正则化的作用，你会发现大量毫无区分度的词汇对应的 $\eta$ 值变成了绝对的 0，这就是所谓的“稀疏性” 。
