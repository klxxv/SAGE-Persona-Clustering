# Inducing Latent Character Personas via Hierarchical Sparse Additive Generative Models (SAGE)

This repository implements a Bayesian generative framework for inducing latent character personas from literary corpora. Drawing inspiration from the work of Bamman et al. (2013, 2014), the model utilizes **Sparse Additive Generative Models (SAGE)** to represent characters as mixtures of latent personas, characterized by their distinctive linguistic deviations from a global background distribution across various dependency roles.

## 1. Theoretical Framework

### 1.1 Persona Induction as Bayesian Inference
Characters in literature are defined not just by their identities, but by the actions they perform (**Agent**), the actions performed upon them (**Patient**), their attributes (**Predicative**), and their belongings (**Possessive**). We model each character $c$ as a categorical distribution over $P$ latent personas:
$$z_c \sim \text{Categorical}(\theta_{book})$$
where $\theta$ is influenced by a Dirichlet prior centered on a book-level or author-level distribution.

### 1.2 Hierarchical SAGE Formulation
Standard Topic Models (like LDA) often struggle with high-frequency stop words and fail to capture subtle semantic deviations. Following Eisenstein et al. (2011), we use SAGE to model the log-linear deviation of word frequencies. The log-likelihood of a word $w$ given a character $c$ and role $r$ is:
$$P(w | c, r) \propto \exp(\eta_{bg, r, w} + \eta_{meta, m, r, w} + \eta_{pers, z_c, r, w})$$
- **$\eta_{bg, r}$**: Global background distribution for role $r$.
- **$\eta_{meta, m}$**: Metadata-specific (e.g., author $m$) deviation.
- **$\eta_{pers, z}$**: Persona-specific deviation for latent persona $z$.

To enforce interpretability and prevent overfitting, we apply a Laplace prior (L1 regularization) on $\eta_{pers}$, forcing non-essential deviations to zero.

### 1.3 Semantic Hierarchy (Balanced Binary Trees)
To mitigate the computational complexity of large vocabularies and incorporate semantic similarity, the model employs a **Hierarchical SAGE** structure. Words are clustered (using Word2Vec or BERT embeddings) into a strictly balanced binary tree. Instead of estimating a flat distribution, the model estimates the probability of "turning left or right" at each internal node of the tree, allowing for:
1. **Parameter Efficiency**: Shared parameters for semantically related words.
2. **Computational Scaling**: $O(\log V)$ complexity per word.

## 2. Key Components

### 2.1 Character Role Features
The model extracts four primary dependency-based features for each character:
- **Agent**: Verbs where the character is the subject.
- **Patient**: Verbs where the character is the object.
- **Possessive**: Nouns possessed by the character.
- **Predicative**: Adjectives describing the character.

### 2.2 Optimization: OWL-QN
Since L1 regularization results in non-differentiable points at zero, we utilize the **Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)** algorithm. This ensures that the persona deviations are truly sparse, highlighting only the most salient linguistic markers for each persona.

## 3. Usage and Pipeline

### 3.1 Training & Grid Search (`traditional_search.py`)
To find the optimal number of personas ($P$) and regularization strength ($\lambda$), use the grid search script. This script orchestrates multiple training runs across different semantic clusterings (BERT vs. Word2Vec).

```bash
python traditional_search.py --start_p 5 --end_p 15 --l1 1.0 --iters 500 --labels W2V-512
```

### 3.2 Evaluation & Keyword Extraction (`traditional_eval.py`)
After training, the evaluation script reconstructs the hierarchical tree and computes the **Total Leaf Effects**:
$$\text{Total Weight} = \eta_{bg} + \eta_{pers}$$
It exports:
1. **Character Assignments**: Maps each character to their most probable latent persona.
2. **Keywords**: Extracts top words per persona/role based on their total SAGE weights.
3. **Filtered Keywords**: Removes "globally distinctive" words that appear in over 80% of personas to ensure persona-specific uniqueness.

```bash
python traditional_eval.py --labels W2V-512 --start_p 8 --end_p 8 --data data/processed/female_words_with_ids.csv
```

## 4. Requirements
- Python 3.8+
- PyTorch (CUDA supported)
- Scikit-learn, Pandas, Numpy
- Joblib (for parallel EM E-steps)

## 5. References
- Eisenstein, J., Ahmed, A., & Xing, E. P. (2011). **Sparse Additive Generative Models of Text.** *ICML*.
- Bamman, D., O'Connor, B., & Smith, N. A. (2013). **Learning Latent Personas of Film Characters.** *ACL*.
- Bamman, D., Underwood, T., & Smith, N. A. (2014). **A Bayesian Mixed Effects Model of Literary Character.** *ACL*.
