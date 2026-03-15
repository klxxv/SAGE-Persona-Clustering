import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# 动态添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sage.model import SAGE_CVAE_Flat
from sage.metrics import calculate_silhouette_custom, calculate_mmd_silhouette

def run_20k_analysis():
    # 1. 配置路径
    word_csv = "fullset_data/word2vec_clusters.csv" 
    bert_csv = "fullset_data/bert_clusters.csv"
    data_file = "fullset_data/all_words.csv"
    meta_file = "data/raw/all_characters_metadata.csv"
    checkpoint_path = "checkpoints/cvae_flat_full/cvae_flat_full_model.pt"
    output_dir = "data/results/full_20k_analysis"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> [Analysis] Starting analysis for top 20,000 characters on {device}")

    # 2. 加载数据
    df_words = pd.read_csv(word_csv)
    vocab = df_words['word'].tolist()
    word_map = {w: i for i, w in enumerate(vocab)}
    V, R = len(vocab), 4

    df_counts = pd.read_csv(data_file)
    df_counts['char_key'] = df_counts['book'] + "_" + df_counts['char_id'].astype(str)
    
    char_freq = df_counts.groupby('char_key')['count'].sum().sort_values(ascending=False)
    top_keys = char_freq.index[:20000].tolist()
    print(f">>> Selected Top {len(top_keys)} characters.")

    # 4. 构建特征矩阵
    df_top = df_counts[df_counts['char_key'].isin(top_keys)].copy()
    df_top['c_idx_local'] = df_top['char_key'].map({k: i for i, k in enumerate(top_keys)})
    df_top['w_idx'] = df_top['word'].map(word_map)
    df_top = df_top.dropna(subset=['w_idx'])
    
    char_feats_np = np.zeros((len(top_keys), V), dtype=np.float32)
    for row in df_top.itertuples():
        char_feats_np[row.c_idx_local, int(row.w_idx)] += row.count
    char_feats_tensor = F.normalize(torch.tensor(char_feats_np), p=2, dim=1).to(device)

    # 5. 推理
    authors = sorted(df_counts["author"].unique())
    model = SAGE_CVAE_Flat(V, len(authors), 8, R).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model.encoder(char_feats_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        labels = torch.argmax(logits, dim=-1).cpu().numpy()
        logits_np = logits.cpu().numpy()

    # 6. 可视化 (采样 5000)
    print(">>> Running t-SNE (Sampled 5000)...")
    viz_idx = np.random.choice(len(logits_np), min(5000, len(logits_np)), replace=False)
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(logits_np[viz_idx])
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42).fit_transform(logits_np[viz_idx])
    
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=labels[viz_idx], cmap='tab10', s=10, alpha=0.6)
    plt.savefig(os.path.join(output_dir, "full_tsne_2d.png"), dpi=300); plt.close()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2], c=labels[viz_idx], cmap='tab10', s=5, alpha=0.5)
    plt.savefig(os.path.join(output_dir, "full_tsne_3d.png"), dpi=300); plt.close()

    # 指标
    def load_v(path):
        df = pd.read_csv(path)
        df['v'] = df['vector'].apply(lambda x: np.array([float(v) for v in x.split(',')]))
        vm = dict(zip(df['word'], df['v']))
        return np.array([vm.get(w, np.zeros(100)) for w in vocab])

    w2v_vecs = load_v(word_csv)
    row_sums = char_feats_np.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1
    char_dists = char_feats_np / row_sums
    
    metrics = {
        "raw_bow": calculate_silhouette_custom(char_dists, labels),
        "latent_logits": calculate_silhouette_custom(logits_np, labels),
        "latent_probs": calculate_silhouette_custom(probs, labels),
        "w2v_weighted": calculate_silhouette_custom(np.dot(char_dists, w2v_vecs), labels),
        "mmd_emd": calculate_mmd_silhouette(char_dists[::25], labels[::25], w2v_vecs)
    }
    
    m_list = sorted([(k, v) for k, v in metrics.items() if isinstance(v, float)], key=lambda x: x[1], reverse=True)
    top_metrics = [m_list[0][0], m_list[1][0]]

    # 角色定义
    persona_interpretations = {
        0: "<b>经典叙事主体</b>：推进情节平稳的核心角色。",
        1: "<b>行动派主角</b>：交互强，具时空移动动能。",
        2: "<b>命运/剧烈冲突型</b>：处于冲突震中，关联死亡与逃亡。",
        3: "<b>现代欲望主体</b>：侧重生活细节与社会资源获取。",
        4: "<b>传统视角观察者</b>：作为见证者或纽带人物。",
        5: "<b>背景/稳定型</b>：具阅历的静态背景角色。",
        6: "<b>高智性内省型</b>：核心动作全是深刻心理活动。",
        7: "<b>敏感感知型主角</b>：侧重细腻的情感感知。"
    }

    df_meta = pd.read_csv(meta_file)
    name_lookup = dict(zip(df_meta['book'] + "_" + df_meta['char_id'].astype(str), df_meta['best_name']))
    
    persona_summary = []
    eta_p = model.decoder.eta_persona.detach().cpu().numpy()
    for p in range(8):
        mask = (labels == p)
        if not any(mask): continue
        kw = {}
        for r_i, r_n in {0:'Agent', 1:'Patient', 2:'Possessive', 3:'Predicative'}.items():
            top_idx = np.argsort(eta_p[p, r_i, :])[-15:][::-1]
            kw[r_n] = [vocab[i] for i in top_idx]
        
        p_chars = [f"{name_lookup.get(top_keys[i], 'Unknown')} ({top_keys[i].split('_')[0]})" for i, v in enumerate(mask) if v][:15]
        persona_summary.append({
            "id": p, "count": int(mask.sum()), "chars": p_chars,
            "keywords": kw, "interpretation": persona_interpretations.get(p, "未定义")
        })

    # HTML 构建 (手动拼接避免语法错误)
    m_html = ""
    for k, v in metrics.items():
        is_top = k in top_metrics
        m_html += '<div class="metric-item ' + ("champion" if is_top else "") + '">'
        m_html += '<div class="metric-val">' + (f"{v:.4f}" if isinstance(v, float) else str(v)) + \
                  (' <span class="champion-badge">TOP</span>' if is_top else "") + '</div>'
        m_html += '<div class="metric-label">' + k.upper() + '</div></div>'

    p_html = ""
    for p in persona_summary:
        kw_rows = ""
        for rn, words in p['keywords'].items():
            kw_rows += f'<div class="role-title">{rn}</div><div class="word-list">{", ".join(words[:10])}</div>'
        
        p_html += f"""<div class="card">
            <div class="persona-id">Cluster {p['id']} <span class="count-tag">{p['count']} Characters</span></div>
            <div class="interpretation">{p['interpretation']}</div>
            <div class="char-list">代表人物: {", ".join(p['chars'])}</div>
            {kw_rows}
        </div>"""

    full_html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>20,000 全角色分析</title><style>
        body { font-family: 'Segoe UI', sans-serif; background: #0a0c12; color: #e4e6ed; padding: 40px; }
        .container { max-width: 1400px; margin: 0 auto; } .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }
        .card { background: #161b26; border: 1px solid #2d333f; border-radius: 14px; padding: 24px; }
        .persona-id { font-size: 24px; font-weight: 700; color: #6c8aff; }
        .interpretation { background: rgba(108,138,255,0.1); border-radius: 8px; padding: 12px; margin-bottom: 15px; font-size: 14px; border-left: 3px solid #6c8aff; }
        .metric-box { background: #11141d; padding: 25px; border-radius: 16px; margin-bottom: 40px; border: 1px solid #2d333f; }
        .metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-top: 15px; }
        .metric-item { background: #161b26; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid transparent; }
        .metric-item.champion { border-color: #4ade80; background: rgba(74,222,128,0.05); } .metric-val { font-size: 18px; color: #4ade80; font-weight: 800; }
        .metric-label { font-size: 10px; color: #8b90a0; margin-top: 5px; } .champion-badge { background: #4ade80; color: #000; font-size: 9px; padding: 2px 5px; border-radius: 4px; }
        .role-title { color: #6c8aff; font-size: 11px; font-weight: 800; text-transform: uppercase; margin-top: 16px; border-bottom: 1px solid #2d333f; }
        .word-list { font-size: 13px; color: #b0b3c1; margin-top: 5px; } .char-list { font-size: 12px; color: #4ade80; font-style: italic; margin-bottom: 15px; }
        img { max-width: 100%; border-radius: 12px; margin-top: 20px; border: 1px solid #2d333f; }
    </style></head><body><div class="container">
        <h1>20,000 全角色人格解耦分析报告 (Full Scale)</h1>
        <div class="metric-box"><h3>多尺度聚类强度看板 (Silhouette Metrics)</h3><div class="metric-grid">""" + m_html + """</div></div>
        <div style="display: flex; gap: 20px; margin-bottom: 40px;"><div style="flex: 1;"><h3>2D 全量角色投影</h3><img src="full_tsne_2d.png"></div>
        <div style="flex: 1;"><h3>3D 拓扑结构</h3><img src="full_tsne_3d.png"></div></div>
        <div class="grid">""" + p_html + """</div></div></body></html>"""
    
    with open(os.path.join(output_dir, "full_20k_analysis_report.html"), "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f">>> Analysis complete. Files saved in {output_dir}")

if __name__ == "__main__":
    run_20k_analysis()
