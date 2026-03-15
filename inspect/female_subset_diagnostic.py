import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch

def diagnostic_female_subset():
    # 1. 路径设置
    data_file = "fullset_data/all_words.csv"
    target_file = "data/processed/target_female_ids.csv"
    output_html = "inspect/female_statistical_report.html"

    print(f">>> [Diagnostic] Loading data...")
    if not os.path.exists(data_file) or not os.path.exists(target_file):
        print("Error: Missing all_words.csv or target_female_ids.csv")
        return

    df_all = pd.read_csv(data_file)
    df_targets = pd.read_csv(target_file)

    # 2. 关联数据
    # 确保 char_id 类型一致
    df_all['char_id'] = df_all['char_id'].astype(int)
    df_targets['char_id'] = df_targets['char_id'].astype(int)
    
    # Normalize book names for fuzzy matching
    def normalize_book(name):
        # Remove underscores, spaces, and lowercase
        return "".join(str(name).replace("_", "").split()).lower()

    df_all['book_norm'] = df_all['book'].apply(normalize_book)
    df_targets['book_norm'] = df_targets['book'].apply(normalize_book)
    
    # Merge on normalized book and char_id
    df_merged = pd.merge(df_all, df_targets, left_on=['book_norm', 'char_id'], right_on=['book_norm', 'char_id'], suffixes=('', '_target'))
    
    # --- CRUCIAL: Aggregate by Name ---
    # Many IDs might belong to one Name in one Book. Merge their word counts.
    print(">>> Aggregating words by character name...")
    df_female = df_merged.groupby(['book', 'name', 'role', 'word'])['count'].sum().reset_index()
    
    # Each character is now a (Book, Name) pair
    df_female['char_key'] = df_female['book'] + "_" + df_female['name']
    char_keys = df_female['char_key'].unique()
    total_females = len(char_keys)
    
    print(f"    Total unique female characters (by name): {total_females}")
    print(f"    (Reduced from {len(df_merged[['book', 'char_id']].drop_duplicates())} raw ID entities)")
    
    roles = ['agent', 'patient', 'possessive', 'predicative']
    stats = []

    # 3. 按维度分析
    for role in roles:
        role_df = df_female[df_female['role'] == role]
        total_tokens = role_df['count'].sum()
        unique_words = role_df['word'].nunique()
        avg_tokens = total_tokens / total_females if total_females > 0 else 0
        
        # 提取 Top 10 关键词
        top_words = role_df.groupby('word')['count'].sum().sort_values(ascending=False).head(10).index.tolist()
        
        # 计算多样性 (Diversity)
        # 针对该维度构建 BoW 矩阵
        role_vocab = role_df['word'].unique()
        v_map = {w: i for i, w in enumerate(role_vocab)}
        c_map = {ck: i for i, ck in enumerate(char_keys)}
        
        diversity = 0
        if len(role_vocab) > 0 and total_females > 1:
            # 采样 500 个角色计算相似度以防内存溢出
            sample_size = min(500, total_females)
            sample_keys = np.random.choice(char_keys, sample_size, replace=False)
            
            feat_matrix = np.zeros((sample_size, len(role_vocab)))
            for r in role_df.itertuples():
                ck = f"{r.book}_{r.name}"
                if ck in sample_keys:
                    s_idx = np.where(sample_keys == ck)[0][0]
                    feat_matrix[s_idx, v_map[r.word]] += r.count
            
            # 归一化后计算余弦相似度
            norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            feat_norm = feat_matrix / norms
            cos_sim = cosine_similarity(feat_norm)
            # Diversity = 1 - 平均两两相似度
            diversity = 1 - (np.sum(cos_sim) - sample_size) / (sample_size * (sample_size - 1))

        stats.append({
            'Role': role,
            'Total_Tokens': total_tokens,
            'Unique_Words': unique_words,
            'Avg_Tokens': avg_tokens,
            'Diversity': diversity,
            'Top_Words': ", ".join(top_words)
        })

    # 4. 生成报告
    rows = ""
    for s in stats:
        rows += f"""<tr>
            <td><b>{s['Role']}</b></td>
            <td>{s['Total_Tokens']}</td>
            <td>{s['Unique_Words']}</td>
            <td>{s['Avg_Tokens']:.2f}</td>
            <td>{s['Diversity']:.4f}</td>
            <td style='font-size:12px'>{s['Top_Words']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>女性角色统计报告</title>
    <style>
        body {{ font-family: sans-serif; padding: 40px; background: #f8f9fa; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; border-bottom: 1px solid #ddd; text-align: left; }}
        th {{ background: #007bff; color: white; }}
        tr:hover {{ background: #f1f1f1; }}
    </style></head><body><div class="container">
        <h1>目标女性角色数据统计报告 (Female Subset Stats)</h1>
        <p>分析对象：<b>{total_females}</b> 个目标实体 (来自 target_female_ids.csv)</p>
        <table><thead><tr>
            <th>维度 (Role)</th><th>总词频</th><th>Unique词数</th><th>人均词数</th><th>多样性</th><th>特征词 (Top 10)</th>
        </tr></thead><tbody>{rows}</tbody></table>
    </div></body></html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f">>> Statistical report saved to {output_html}")

if __name__ == "__main__":
    diagnostic_female_subset()
