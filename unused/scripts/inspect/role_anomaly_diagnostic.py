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
    data_file = "fullset_data/all_words.csv"
    checkpoint_path = "checkpoints/cvae_flat_full/cvae_flat_full_model.pt"
    output_html = "inspect/role_anomaly_report.html"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> [Diagnostic] Starting deep role analysis on {device}")

    # 2. 加载数据
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run extraction first.")
        return
    
    df = pd.read_csv(data_file)
    
    if not os.path.exists(word_csv):
        print(f"Warning: {word_csv} not found. Sensitivity analysis will be limited.")
        # Fallback: simple vocab from data
        vocab = sorted(df['word'].unique().tolist())
    else:
        df_words = pd.read_csv(word_csv)
        vocab = df_words['word'].tolist()
        
    word_map = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    df['char_key'] = df['book'] + "_" + df['char_id'].astype(str)
    # 强制指定顺序以匹配模型习惯
    roles = ['agent', 'patient', 'possessive', 'predicative']
    available_roles = [r for r in roles if r in df['role'].unique()]
    
    char_keys = sorted(df['char_key'].unique())
    C = len(char_keys)
    char_to_idx = {ck: i for i, ck in enumerate(char_keys)}

    # 3. 尝试加载 CVAE 模型
    model = None
    if os.path.exists(checkpoint_path):
        try:
            M = len(df['author'].unique())
            model = SAGE_CVAE_Flat(V, M, 8, 4).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            print(f">>> Found model at {checkpoint_path}, enabling sensitivity analysis.")
        except Exception as e:
            print(f">>> Failed to load model: {e}. Running data-only diagnostic.")
    else:
        print(f">>> No model found at {checkpoint_path}. Running data-only diagnostic.")

    # 采样 1000 个角色用于耗时的距离计算
    sample_size = min(1000, C)
    sample_keys = np.random.choice(char_keys, sample_size, replace=False)
    sample_indices = [char_to_idx[k] for k in sample_keys]

    stats = []
    for role in roles:
        if role not in available_roles:
            print(f">>> Role: {role} is MISSING in data!")
            stats.append({
                "Role": role, "Total_Tokens": 0, "Unique_Words": 0,
                "Avg_Tokens": 0, "Diversity": 0, "Sensitivity": 0,
                "Entropy": 0, "Diagnosis": "<b style='color:red'>缺失数据</b>"
            })
            continue

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
        
        logits_std = 0
        entropy = 0
        if model is not None:
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
        if diversity < 0.05: diagnosis = "<b>描写同质化</b>：用词在角色间高度重合。"
        elif model is not None and logits_std < 0.5: diagnosis = "<b>特征盲区</b>：Encoder 对此维度不敏感。"
        elif tokens_per_char < 0.5: diagnosis = "<b>信息极度稀疏</b>。"

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
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f4f7f6; padding: 40px; line-height: 1.6; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #6c8aff; padding-bottom: 10px; margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }} 
        th, td {{ padding: 15px; border-bottom: 1px solid #eee; text-align: left; }}
        th {{ background: #6c8aff; color: white; font-size: 13px; text-transform: uppercase; }}
        .highlight {{ color: #e74c3c; font-weight: bold; }} .good {{ color: #27ae60; font-weight: bold; }}
        .formula-section {{ background: #1e222d; color: #e4e6ed; padding: 25px; border-radius: 12px; margin-top: 40px; font-family: 'Consolas', monospace; }}
        .formula-section h3 {{ color: #6c8aff; margin-top: 0; font-family: sans-serif; }}
        .formula-item {{ margin-bottom: 15px; border-bottom: 1px solid #2d333f; padding-bottom: 10px; }}
        .formula-title {{ color: #4ade80; font-weight: bold; margin-bottom: 5px; display: block; }}
    </style></head><body><div class="container">
        <h1>SAGE 依存维度异常深度诊断 (Multi-Dimensional Diagnostic)</h1>
        
        <table><thead><tr>
            <th>Role</th><th>总词频</th><th>Unique词数</th><th>Avg词数/角色</th><th>多样性 (Diversity)</th>
            <th>敏感度 (Sensitivity)</th><th>分配熵 (Entropy)</th><th>AI 自动诊断</th>
        </tr></thead><tbody>{rows}</tbody></table>

        <div class="formula-section">
            <h3>📊 指标计算公式说明 (Methodology)</h3>
            
            <div class="formula-item">
                <span class="formula-title">1. 词袋多样性 (Diversity Score)</span>
                <code>Diversity = 1 - [ Σ cos_sim(x_i, x_j) - N ] / [ N * (N - 1) ]</code><br>
                <small>计算角色在特定维度下 BoW 向量的平均两两余弦相似度。越接近 1 说明角色间的该维度描写越迥异，越接近 0 说明极其趋同。</small>
            </div>

            <div class="formula-item">
                <span class="formula-title">2. Encoder 敏感度 (Sensitivity Std)</span>
                <code>Sensitivity = mean( std( Encoder_Logits(x)_axis=0 ) )</code><br>
                <small>将特定维度的特征单独输入 Encoder，计算输出 Logits 在不同角色间的标准差。高 Std 意味着 Encoder 对该维度的输入信号有剧烈反应。</small>
            </div>

            <div class="formula-item">
                <span class="formula-title">3. 人格分配熵 (Persona Entropy)</span>
                <code>Entropy = - Σ [ P(z|x) * log(P(z|x)) ]</code><br>
                <small>基于 Shannon Entropy。高熵意味着模型对该维度的判断模棱两可；低熵（接近 0）意味着该维度提供了极具确定性的人格分类信号。</small>
            </div>

            <div class="formula-item">
                <span class="formula-title">4. 角色平均词数 (Avg Tokens)</span>
                <code>Avg_Tokens = Total_Tokens_in_Role / Total_Characters</code><br>
                <small>量化该维度的信息密度。如果 Avg &lt; 1，说明该维度在大多数角色身上是缺失的。</small>
            </div>
        </div>
    </div></body></html>"""
    
    with open(output_html, "w", encoding="utf-8") as f: f.write(html)
    print(f">>> Report saved to {output_html}")

if __name__ == "__main__":
    diagnostic_role_variance()
