# Beta-VAE 模型架构总结

本文档总结了 `sage/beta_vae_model.py` 中实现的 `SAGE_BetaVAE` 模型的输入、输出、中间层架构、损失函数定义以及类别分离（解耦）机制。

## 1. 输入特征

*   **连续特征 (`char_feats`)**：基于 TF-IDF 或词频的词袋模型（BoW），表示角色在不同语法角色下的词汇分布。形状：`[C, V]`，其中 `C` 为角色总数，`V` 为词汇表大小。在 `BetaLiterarySAGE.fit()` 中，该输入会进行 L2 归一化。
*   **索引（Indices）**：
    *   `m_idx`：作者索引（用于控制作者特定的背景风格）。
    *   `c_idx`：角色索引（用于获取对应的 `char_feats`）。
    *   `r_idx`：语法角色索引（例如：agent/施事者，patient/受事者等）。
    *   `w_idx`：目标词汇索引，用于计算重建损失。
    *   `count`：词汇频次权重，用于损失计算中的加权。

## 2. 模型架构 (中间层)

该架构建立在标准的 VAE 框架之上，为了实现连续特征的解耦（Disentanglement）而修改为 Beta-VAE。

### 编码器 (Encoder): `CharacterContinuousEncoder`
*   **输入**：`char_feats`（每个角色维度为 `V`）。
*   **网络结构**：
    *   全连接层 (`fc1`)：从维度 `V` 映射到隐藏层 `hidden_dim`（默认 512）。
    *   层归一化 (`bn1`) + ReLU 激活函数 + Dropout (p=0.2)。
    *   两个平行的输出层：
        *   `fc_mu`：`hidden_dim` -> `K`（潜空间维度，即连续特征轴的数量）。
        *   `fc_logvar`：`hidden_dim` -> `K`（潜空间正态分布的对数方差）。
*   **输出**：$\mu$ (均值) 和 $\log(\sigma^2)$ (对数方差)，它们代表了该角色在 $K$ 维连续潜空间中的正态分布参数。

### 重参数化技巧 (Reparameterization Trick)
*   在训练期间，从 $\mathcal{N}(\mu, \sigma^2)$ 中采样连续的向量 $z$：$z = \mu + \epsilon \cdot e^{0.5 \cdot \text{logvar}}$，其中 $\epsilon \sim \mathcal{N}(0, I)$。
*   在推理/预测阶段，直接使用均值 $z = \mu$。

### 解码器 (Decoder): `FlatContinuousDecoder`
*   **生成参数 (Generative Parameters)**：
    *   `eta_bg`：基础的全局词汇对数几率。形状：`[R, V]`。
    *   `eta_author`：作者特定的词汇对数几率偏移。形状：`[M, R, V]`。
    *   **`eta_axes`**：潜在特征轴，代表相互独立的语义或风格维度。形状：`[K, R, V]`。
*   **前向传播 (Forward Pass)**：
    *   给定输入 $z \in \mathbb{R}^{B \times K}$（其中 $B$ 是 batch size），模型将这 $K$ 个特征轴进行线性组合：$z \times \text{eta\_axes}$。这产生了一个受潜变量影响的词汇分布偏置，形状为 `[B, R, V]`。
    *   提取与当前对应的 `r_idx` 那一行的偏置：`axes_w = sampled_axes[batch_indices, r_idx]`。
    *   **Logits 计算**：$\text{Logits} = \text{eta\_bg}[r] + \text{eta\_author}[m, r] + \text{axes\_w}$
    *   **最终输出**：通过 `F.log_softmax(logits, dim=-1)` 得到预测词汇的对数概率。

## 3. 损失函数定义 (Loss Definitions)

训练损失是三个部分的加权和。其中 KL 散度的权重 $\beta$ 对模型行为影响巨大（这也是其被称为 Beta-VAE 的原因）。

`Loss = recon_loss + beta * kl_loss + l1_lambda * l1_loss`

1.  **重建损失 (Reconstruction Loss / NLL)**：
    *   负对数似然损失，衡量在给定输入条件下，预测真实词汇 `w_idx` 的准确度。
    *   `recon_loss = -torch.sum(log_probs[batch, b_w] * b_count) / b_count.sum()`
    *   它代表了模型从连续的潜变量 $z$ 中重构出原始“角色-词汇分布”的能力。

2.  **KL 散度 (KL Divergence, `kl_loss`)**：
    *   迫使编码器输出的连续隐变量分布 $q(z|x) = \mathcal{N}(\mu, \sigma^2)$ 逼近标准正态先验分布 $p(z) = \mathcal{N}(0, I)$。
    *   `kl_loss = -0.5 * torch.sum(1 + logvar - mu^2 - exp(logvar)) / batch_size`
    *   **`beta` 权重（默认 2.0）鼓励强解耦（Disentanglement）**。较大的 `beta` 会对复杂的潜变量表示施加重罚，迫使模型只能把最重要、最相互正交（不相关）的特征编码到这 $K$ 个轴里。

3.  **L1 正则化 (`l1_loss`)**：
    *   鼓励生成权重参数的稀疏性。它确保每个特征轴 (`eta_axes`) 和每个作者的风格 (`eta_author`) 只由一小部分最具代表性的核心词汇定义，而不是泛泛地影响所有词汇。
    *   `l1_loss = sum(|eta_author|) + sum(|eta_axes|)`

## 4. 输出与类别分离 (解耦机制 / Disentanglement)

与强制将角色硬分配到离散的类别/人格（如 CVAE 和传统 GMM 模型中的 Softmax 机制）不同，Beta-VAE 将角色映射到了一个**连续的 $K$ 维空间**中。

*   **输出表示**：每个角色的最终“人格”表示为一个坐标向量 $z \in \mathbb{R}^K$。
*   **类别分离机制 (语义连续谱)**：
    *   模型不再学习互斥的聚类（Clusters），而是学习了 $K$ 条相互独立的**连续特征谱/轴（Spectrums/Axes）**。
    *   得益于高强度的 $\beta$ KL 惩罚，向量 $z$ 中的各个维度（即 $K$ 个轴）在统计上被迫互相独立。
    *   每个轴 $k$ 关联一个生成矩阵 $\text{eta\_axes}[k]$。如果角色的 $z_k$ 值是正的，那么在 $\text{eta\_axes}[k]$ 中正权重的词概率就会增加；反之，负的 $z_k$ 会抑制这些词，并提升负权重的词。
    *   **提取特征**：通过检查 `eta_axes[k, r, :]` 中的最大正权重和最小负权重，我们可以提取出该特征轴的**“对立两极”（Opposite poles）**（例如，一个轴的正极可能代表“暴力/血腥”，负极代表“和平/温和”；另一个轴可能代表“富裕”与“贫穷”）。
    *   **结果解释**：角色不再被“分配”给某一个单一的人格标签。相反，每个角色就像在玩 RPG 游戏时调整的“属性滑块”一样，分布在这 $K$ 个独立性格维度的坐标空间中。