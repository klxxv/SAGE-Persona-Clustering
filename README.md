# Inducing Latent Character Personas via Hierarchical Sparse Additive Generative Models (SAGE)
# 基于层次稀疏加性生成模型（SAGE）的文学角色潜在人格归纳

This repository implements a Bayesian generative framework for inducing latent character personas from literary corpora. Drawing inspiration from the work of Bamman et al. (2013, 2014), the model utilizes **Sparse Additive Generative Models (SAGE)** to represent characters as mixtures of latent personas, characterized by their distinctive linguistic deviations from a global background distribution across various dependency roles.

本仓库实现了一套贝叶斯生成框架，用于从文学语料库中归纳角色的潜在人格。受 Bamman et al.（2013, 2014）工作的启发，模型采用**稀疏加性生成模型（SAGE）**将角色表示为若干潜在人格的混合，以各依存角色下相对全局背景分布的语言偏差为特征。

---

## 1. Theoretical Framework / 理论框架

### 1.1 Persona Induction as Bayesian Inference / 人格归纳的贝叶斯视角

Characters in literature are defined not just by their identities, but by the actions they perform (**Agent**), the actions performed upon them (**Patient**), their attributes (**Predicative**), and their belongings (**Possessive**). We model each character $c$ as a categorical distribution over $P$ latent personas:

文学角色不仅由身份界定，还由其**施事（Agent）**动作、**受事（Patient）**动作、**描述性属性（Predicative）**和**领属关系（Possessive）**共同刻画。我们将每个角色 $c$ 建模为 $P$ 个潜在人格上的类别分布：

$$z_c \sim \text{Categorical}(\theta_{book})$$

where $\theta$ is influenced by a Dirichlet prior centered on a book-level or author-level distribution.

其中 $\theta$ 由以书籍或作者层级分布为中心的 Dirichlet 先验约束。

### 1.2 Hierarchical SAGE Formulation / 层次 SAGE 形式化

Standard Topic Models (like LDA) often struggle with high-frequency stop words and fail to capture subtle semantic deviations. Following Eisenstein et al. (2011), we use SAGE to model the log-linear deviation of word frequencies. The log-likelihood of a word $w$ given a character $c$ and role $r$ is:

标准主题模型（如 LDA）在处理高频停用词时往往力不从心，也难以捕捉细微的语义偏差。遵循 Eisenstein et al.（2011）的思路，我们用 SAGE 对词频的对数线性偏差建模。给定角色 $c$ 和依存角色 $r$，词 $w$ 的对数似然为：

$$P(w | c, r) \propto \exp(\eta_{bg, r, w} + \eta_{meta, m, r, w} + \eta_{pers, z_c, r, w})$$

- **$\eta_{bg, r}$**: Global background distribution for role $r$. / 角色 $r$ 的全局背景分布。
- **$\eta_{meta, m}$**: Metadata-specific (e.g., author $m$) deviation. / 元数据（如作者 $m$）的偏差。
- **$\eta_{pers, z}$**: Persona-specific deviation for latent persona $z$. / 潜在人格 $z$ 的特定偏差。

To enforce interpretability and prevent overfitting, we apply a Laplace prior (L1 regularization) on $\eta_{pers}$, forcing non-essential deviations to zero.

为增强可解释性并防止过拟合，我们对 $\eta_{pers}$ 施加 Laplace 先验（L1 正则化），迫使非关键偏差归零。

### 1.3 Semantic Hierarchy (Two-Stage Cluster Tree) / 语义层次结构（两阶段聚类树）

To mitigate the computational complexity of large vocabularies and incorporate semantic similarity, the model employs a **two-stage Hierarchical SAGE** structure built over Word2Vec embeddings:

为降低大词表的计算复杂度并融入语义相似性，模型采用基于 Word2Vec 嵌入的**两阶段层次 SAGE** 结构：

**Stage 1 — K-Means vocabulary compression / 第一阶段：K-Means 词表压缩** (`preprocess/build_role_clusters.py`): Each role's vocabulary is independently compressed by K-Means into $K = 2^{\lfloor\log_2(V/10)\rceil}$ clusters (nearest power of 2 to $V/10$). This yields a compact cluster vocabulary per role, with each word mapped to its cluster centroid.

每个角色的词表独立经 K-Means 压缩为 $K = 2^{\lfloor\log_2(V/10)\rceil}$ 个聚类（$V/10$ 的最近 2 的幂次）。每个词映射到其所属聚类的质心，形成各角色独立的紧凑聚类词表。

**Stage 2 — Greedy nearest-neighbor tree / 第二阶段：贪心最近邻建树** (`sage/model_traditional.py`): The model then constructs a **greedy agglomerative binary tree** over the $K$ cluster centroids. At each level, pairs of nodes with the smallest Euclidean distance between their centroid vectors are merged bottom-up. The result is a tree whose **leaf nodes are clusters** (not individual words) — all words sharing a cluster traverse the same path.

模型随后对 $K$ 个聚类质心构建**贪心自底向上二叉树**：每层将欧氏距离最近的质心对两两合并，父节点取子节点均值。**树的叶节点是聚类而非单词**——同一聚类内的所有词共享相同的树路径。

> **Note / 注意**: Unlike a strictly balanced binary tree, the greedy pairing can produce uneven depth across branches. Choosing $K$ as a power of 2 keeps the tree near-balanced in practice.
>
> 与严格平衡二叉树不同，贪心配对可能导致各分支深度不一。将 $K$ 取为 2 的幂次可在实践中使树接近平衡。

This two-stage design allows for: / 该两阶段设计实现了：

1. **Parameter Efficiency / 参数高效**: Multiple synonymous words share a single leaf node. / 语义相近词共享同一叶节点的参数。
2. **Semantic Structure / 语义结构**: Tree topology reflects Word2Vec proximity. / 树拓扑反映 Word2Vec 语义邻近性。
3. **Computational Scaling / 计算可扩展**: $O(\log K)$ complexity per observation, with $K \ll V$. / 每条观测 $O(\log K)$ 复杂度，且 $K \ll V$。

---

## 2. Key Components / 核心组件

### 2.1 Character Role Features / 角色依存特征

The model extracts four primary dependency-based features for each character:

模型为每个角色提取四类主要依存特征：

- **Agent**: Verbs where the character is the subject. / 角色作为主语的动词。
- **Patient**: Verbs where the character is the宾语 (object). / 角色作为宾语的动词。
- **Possessive**: Nouns possessed by the character. / 角色领属的名词。
- **Predicative**: Adjectives describing the character. / 描述角色的形容词。

### 2.2 Optimization: OWL-QN / 优化算法：OWL-QN

Since L1 regularization results in non-differentiable points at zero, we utilize the **Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)** algorithm. This ensures that the persona deviations are truly sparse, highlighting only the most salient linguistic markers for each persona.

由于 L1 正则化在零点产生不可微问题，我们采用**分象限有限内存拟牛顿（OWL-QN）**算法，确保人格偏差真正稀疏，仅保留每个人格最显著的语言标记。

---

## 3. Usage and Pipeline / 使用流程

### 3.1 Preprocessing: Per-Role Cluster Building / 预处理：按角色构建聚类（`preprocess/build_role_clusters.py`）

Before training, each role's vocabulary must be independently clustered. The script reads Word2Vec embeddings, computes per-role K-Means clusters, and saves two files per role:

训练前，需对每个角色的词表独立进行聚类。脚本读取 Word2Vec 嵌入，为每个角色计算 K-Means 聚类，并为每个角色保存两份文件：

| File | Contents | Tracked in git |
|---|---|---|
| `{role}_clusters.csv` | `word`, `cluster_id` only | Yes / 是 |
| `{role}_clusters_full.csv` | `word`, `cluster_id`, `vector` (centroid) | No (gitignored) / 否 |

The cluster count formula is: / 聚类数公式为：

$$K = 2^{\text{round}(\log_2(V_{\text{role}} / 10))}$$

where $V_{\text{role}}$ is the number of unique words for that role. If the full CSV (with vectors) is absent at training time, `_hydrate_cluster_csv()` in `sage/model_traditional.py` recomputes centroids on-the-fly from raw W2V embeddings.

其中 $V_{\text{role}}$ 为该角色的唯一词数。若训练时缺少含向量的完整 CSV，`sage/model_traditional.py` 中的 `_hydrate_cluster_csv()` 会自动从原始 W2V 嵌入实时重算质心。

```bash
python preprocess/build_role_clusters.py
```

### 3.3 Training & Grid Search / 训练与网格搜索（`traditional_search.py`）

To find the optimal number of personas ($P$) and regularization strength ($\lambda$), use the grid search script. This script orchestrates multiple training runs across different semantic clusterings (BERT vs. Word2Vec).

使用网格搜索脚本寻找最优人格数（$P$）和正则化强度（$\lambda$），支持跨不同语义聚类（BERT vs. Word2Vec）的多次训练。

```bash
python traditional_search.py --start_p 5 --end_p 15 --l1 1.0 --iters 500 --labels W2V-512
```

### 3.4 Evaluation & Keyword Extraction / 评估与关键词提取（`traditional_eval.py`）

After training, the evaluation script reconstructs the hierarchical tree and computes the **Total Leaf Effects**:

训练完成后，评估脚本重建层次树并计算**叶节点总效应**：

$$\text{Total Weight} = \eta_{bg} + \eta_{pers}$$

It exports: / 输出内容：
1. **Character Assignments / 角色分配**: Maps each character to their most probable latent persona. / 将每个角色映射至最可能的潜在人格。
2. **Keywords / 关键词**: Extracts top words per persona/role based on their total SAGE weights. / 按 SAGE 总权重提取每个人格/角色的关键词。
3. **Filtered Keywords / 过滤后关键词**: Removes "globally distinctive" words that appear in over 80% of personas to ensure persona-specific uniqueness. / 去除出现在 80% 以上人格中的"全局显著词"，确保人格特异性。

```bash
python traditional_eval.py --labels W2V-512 --start_p 8 --end_p 8 --data data/processed/female_words_with_ids.csv
```

### 3.5 Persona Log-Likelihood Analysis / 人格对数似然分析（`postprocess/persona_loglikelihood.py`）

After persona assignments are obtained, this postprocessing script computes **per-persona distinctive word lists** using **log-likelihood ratio (LLR)** and **Dunning's G² statistic** in a one-vs-rest framework.

在获得人格分配后，此后处理脚本采用**对数似然比（LLR）**和 **Dunning G² 统计量**，以"当前人格 vs. 其余所有人格"的框架，为每个人格计算**显著特征词列表**。

**Method / 方法：**

For each persona $p$, treat its word frequencies as corpus A and all other personas combined as corpus B. With Laplace smoothing ($k=0.5$):

对于人格 $p$，将其词频视为语料库 A，其余所有人格合并为语料库 B，应用 Laplace 平滑（$k=0.5$）：

$$\text{LLR}(w, p) = \log P(w \mid A) - \log P(w \mid B)$$

$$G^2 = 2\left(a \ln\frac{a}{E_1} + b \ln\frac{b}{E_2}\right), \quad E_i = N_i \cdot \frac{a+b}{N_1+N_2}$$

The sign of $G^2$ indicates overrepresentation ($+$) or underrepresentation ($-$) in persona $p$.

$G^2$ 的符号表示该词在人格 $p$ 中是过度表达（$+$）还是不足表达（$-$）。

**Two rankings per (persona, role) / 每个（人格, 角色）对的两种排名：**
1. **Unweighted / 无权重**: Ranked by $G^2$ directly (empirical frequency only). / 直接按 $G^2$ 排序（仅基于经验频率）。
2. **Eta-weighted / eta 加权**: Score $= \eta_{\text{norm}}(p, r, w) \times G^2$, incorporating model-learned importance. / 得分 $= \eta_{\text{norm}}(p, r, w) \times G^2$，融合模型学到的重要性权重。

**Outputs / 输出文件** (saved to `data/results/traditional_results/W2V-Role/P8_L1.0/`):

| File | Description / 描述 |
|---|---|
| `llr_unweighted.csv` | Full LLR + G² scores for all (persona, role, word) triples / 所有三元组的完整 LLR+G² 得分 |
| `llr_eta_weighted.csv` | Full table with eta-weighted scores / 含 eta 加权得分的完整表 |
| `topK_unweighted.csv` | Top-20 words per (persona, role) by G² / 按 G² 每（人格, 角色）前20词 |
| `topK_eta_weighted.csv` | Top-20 words per (persona, role) by eta×G² (positive G² only) / 按 eta×G² 前20词（仅正值） |

```bash
python postprocess/persona_loglikelihood.py
```

---

## 4. Requirements / 依赖环境

- Python 3.8+
- PyTorch (CUDA supported / 支持 CUDA)
- Scikit-learn, Pandas, Numpy
- Joblib (for parallel EM E-steps / 用于并行 EM E 步)

---

## 5. References / 参考文献

- Eisenstein, J., Ahmed, A., & Xing, E. P. (2011). **Sparse Additive Generative Models of Text.** *ICML*.
- Bamman, D., O'Connor, B., & Smith, N. A. (2013). **Learning Latent Personas of Film Characters.** *ACL*.
- Bamman, D., Underwood, T., & Smith, N. A. (2014). **A Bayesian Mixed Effects Model of Literary Character.** *ACL*.
