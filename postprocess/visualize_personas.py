import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import argparse

def plot_persona_visualization(args):
    # 1. 加载 SAGE 预测结果
    df_persona = pd.read_csv(os.path.join(args.model_dir, "sage_character_personas.csv"))
    
    # 2. 加载原始词频数据以获得分布
    df_words = pd.read_csv(args.data_file)
    word_to_cluster = pd.read_csv(args.word_csv_file)
    
    # 计算每个角色的词分布 [C, 1000]
    df_merged = df_words.merge(word_to_cluster[['word', 'cluster_id']], on='word')
    char_counts = df_merged.groupby(['book', 'char_id', 'cluster_id'])['count'].sum().unstack(fill_value=0)
    char_dist = char_counts.div(char_counts.sum(axis=1), axis=0)
    
    # 对齐 Persona 标签
    char_dist = char_dist.reset_index()
    char_dist['char_key'] = char_dist['book'] + "_" + char_dist['char_id'].astype(str)
    df_viz = char_dist.merge(df_persona[['char_key', 'persona']], on='char_key')
    
    # 3. t-SNE 降维
    print(">>> Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # 词簇列为 0-999
    cluster_cols = [c for c in df_viz.columns if isinstance(c, int)]
    if not cluster_cols: # 兜底逻辑：如果列名是字符串
        cluster_cols = [c for c in df_viz.columns if str(c).isdigit()]
        
    X_embedded = tsne.fit_transform(df_viz[cluster_cols].values)
    
    # 4. 绘图
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1], 
        hue=df_viz['persona'], palette='viridis', 
        alpha=0.6, s=40, edgecolor=None
    )
    
    # 5. 计算并标注每个 Persona 的 Top-K 词簇代表词
    # 我们加载词簇代表词：离词簇中心 L1 最近的 token
    # 预计算：每个词簇内找代表词
    def get_representative_token(cluster_id):
        cluster_data = word_to_cluster[word_to_cluster['cluster_id'] == cluster_id].copy()
        if cluster_data.empty: return f"C{cluster_id}"
        # 计算 L1 距离 (向量均值 vs 每个词向量)
        vectors = np.vstack(cluster_data['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')])).values)
        centroid = vectors.mean(axis=0)
        dists = np.sum(np.abs(vectors - centroid), axis=1)
        return cluster_data.iloc[np.argmin(dists)]['word']

    # 计算每个 Persona 的核心词簇
    # 基于 model weights (eta_pers)
    state_dict = torch.load(os.path.join(args.model_dir, "sage_model_weights.pt"), map_location='cpu')
    eta_pers = state_dict['eta_pers'] # [P, R, V+1]
    
    # 取所有角色的平均权重
    persona_weights = eta_pers.mean(dim=1).numpy()[:, :-1] # [P, 1000]
    
    for p in range(len(persona_weights)):
        # 该 Persona 的中心位置 (t-SNE 均值)
        p_mask = df_viz['persona'] == p
        if not p_mask.any(): continue
        center_coords = X_embedded[p_mask].mean(axis=0)
        
        # 寻找 top 16 个最显著的词簇 (4x4)
        top_clusters = np.argsort(persona_weights[p])[-16:][::-1]
        top_tokens = [get_representative_token(cid) for cid in top_clusters]
        
        # 整理成 4x4 文本
        token_grid = "\n".join([", ".join(top_tokens[i:i+4]) for i in range(0, 16, 4)])
        
        plt.text(
            center_coords[0], center_coords[1], 
            f"Persona {p}\n{token_grid}",
            fontsize=8, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
        )

    # 标注指标 (如果有)
    if args.silhouette and args.perplexity:
        plt.title(f"SAGE Persona Clustering (t-SNE)\nSilhouette: {args.silhouette:.4f} | Perplexity: {args.perplexity:.4f}", fontsize=15)
    else:
        plt.title("SAGE Persona Clustering Visualization (t-SNE)", fontsize=15)
        
    plt.tight_layout()
    plt.savefig(args.output_img, dpi=300)
    print(f">>> Visualization saved to {args.output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--word_csv_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_img', type=str, default="persona_viz.png")
    parser.add_argument('--silhouette', type=float)
    parser.add_argument('--perplexity', type=float)
    args = parser.parse_args()
    plot_persona_visualization(args)
