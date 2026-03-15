import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# 动态添加根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sage.model import SAGE_CVAE_Flat

def diagnostic_role_variance():
    # 1. 路径设置
    word_csv = "fullset_data/word2vec_clusters.csv" 
    data_file = "data/processed/all_words.csv"
    checkpoint_path = "checkpoints/cvae_flat_full/cvae_flat_full_model.pt"
    output_html = "inspect/role_anomaly_report.html"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> [Diagnostic] Starting deep role analysis on {device}")

    # 2. 加载数据
    df = pd.read_csv(data_file)
    df_words = pd.read_csv(word_csv)
    vocab = df_words['word'].tolist()
    word_map = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    df['char_key'] = df['book'] + "_" + df['char_id'].astype(str)
    roles = sorted(df['role'].unique().tolist())
    char_keys = sorted(df['char_key'].unique())
    C = len(char_keys)
    char_to_idx = {ck: i for i, ck in enumerate(char_keys)}

    # 3. 加载 CVAE 模型
    M = len(df['author'].unique())
    model = SAGE_CVAE_Flat(V, M, 8, 4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 采样 1000 个角色用于耗时的距离计算
    sample_size = 1000
    sample_keys = np.random.choice(char_keys, sample_size, replace=False)
    sample_indices = [char_to_idx[k] for k in sample_keys]

    stats = []
    for role in roles:
        print(f">>> Analyzing Role: {role}...")
        role_df = df[df['role'] == role].copy()
        total_tokens = role_df['count'].sum()
        unique_words = role_df['word'].nunique()
        tokens_per_char = total_tokens / C
        
        role_feats = np.zeros((C, V), dtype=np.float32)
        role_df['w_idx'] = role_df['word'].map(word_map)
        role_df = role_df.dropna(subset=['w_idx'])
        for r in role_df.itertuples():
            role_feats[char_to_idx[r.char_key], int(r.w_idx)] += r.count
        
        role_feats_norm = F.normalize(torch.tensor(role_feats), p=2, dim=1)
        
        with torch.no_grad():
            sample_feats = role_feats_norm[sample_indices].to(device)
            logits = model.encoder(sample_feats)
            logits_std = torch.std(logits, dim=0).mean().item()
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean().item()

        sample_feats_np = role_feats_norm[sample_indices].cpu().numpy()
        cos_sim_matrix = cosine_similarity(sample_feats_np)
        diversity = 1 - (np.sum(cos_sim_matrix) - sample_size) / (sample_size * (sample_size - 1))

        # 诊断原因
        diagnosis = "正常维度"
        if diversity < 0.05: diagnosis = "<b>描写同质化</b>：用词在角色间高度重合（如手、脸），无区分度。"
        elif logits_std < 0.5: diagnosis = "<b>特征盲区</b>：Encoder 对此维度不敏感。"
        elif tokens_per_char < 1.0: diagnosis = "<b>信息极度稀疏</b>。"

        stats.append({
            "Role": role, "Total_Tokens": int(total_tokens), "Unique_Words": unique_words,
            "Avg_Tokens": tokens_per_char, "Diversity": diversity, "Sensitivity": logits_std,
            "Entropy": entropy, "Diagnosis": diagnosis
        })

    # 5. 生成 HTML
    rows = ""
    for s in stats:
        div_class = "highlight" if s['Diversity'] < 0.1 else "good"
        sen_class = "highlight" if s['Sensitivity'] < 0.8 else "good"
        rows += f"""<tr>
            <td><b>{s['Role']}</b></td><td>{s['Total_Tokens']}</td><td>{s['Unique_Words']}</td>
            <td>{s['Avg_Tokens']:.2f}</td><td class="{div_class}">{s['Diversity']:.4f}</td>
            <td class="{sen_class}">{s['Sensitivity']:.4f}</td><td>{s['Entropy']:.4f}</td><td>{s['Diagnosis']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>SAGE 异常分析报告</title><style>
        body {{ font-family: sans-serif; background: #f4f7f6; padding: 40px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; }}
        table {{ width: 100%; border-collapse: collapse; }} th, td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        th {{ background: #6c8aff; color: white; }} .highlight {{ color: red; font-weight: bold; }} .good {{ color: green; }}
    </style></head><body><div class="container"><h1>SAGE 依存维度异常诊断</h1><table><thead><tr>
        <th>Role</th><th>总词频</th><th>Unique词数</th><th>Avg词数/角色</th><th>多样性 (Diversity)</th>
        <th>敏感度 (Sensitivity)</th><th>分配熵 (Entropy)</th><th>AI 自动诊断</th>
    </tr></thead><tbody>{rows}</tbody></table></div></body></html>"""
    
    with open(output_html, "w", encoding="utf-8") as f: f.write(html)
    print(f">>> Report saved to {output_html}")

if __name__ == "__main__":
    diagnostic_role_variance()
