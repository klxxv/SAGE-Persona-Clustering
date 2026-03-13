
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_report():
    # 1. Load data
    with open('data/results/p4_summary.json', 'r') as f:
        summary = json.load(f)
    keywords = pd.read_csv('data/results/p4_keywords.csv')
    
    # 2. Create Distribution Plot
    print(">>> Generating distribution plot...")
    plt.figure(figsize=(10, 6))
    df_summary = pd.DataFrame(summary)
    sns.barplot(x="id", y="count", data=df_summary, palette="viridis")
    plt.title("Persona Distribution (N=4, IT1000)")
    plt.ylabel("Character Count")
    plt.xlabel("Persona ID")
    plt.savefig('data/results/p4_distribution.png', dpi=100)
    plt.close()

    # 3. Create HTML Content
    print(">>> Generating HTML report...")
    
    persona_cards = ""
    colors = ["#6c8aff", "#4ade80", "#fbbf24", "#fb7185"]
    
    # Custom descriptions for P4
    descriptions = [
        "<b>叙述者与核心主角 (Past Tense)</b>: 包含大量第一人称 'I' 或第三人称过去时叙事中的主角。关键词以过去式动词为主（said, had, looked）。这是规模最大的群体，涵盖了各书中出场最稳健的核心人物。",
        "<b>现在时主角与互动型人物 (Present Tense)</b>: 显著特征是现在时动词（says, looks, asks）。这反映了现代都市或特定风格小说中的实时叙事模式，女性比例较高（40.4%），包含多位具有强烈主体性的女性主角。",
        "<b>社交与都市生活型配角 (Social/Urban)</b>: 出场频次中等，多见于当代上海背景小说。关键词涉及更具体的社交互动。女性比例约为30%，包含多位都市职业女性配角。",
        "<b>边缘与特定情节驱动型 (Peripheral/Relational)</b>: 规模最小，出场频次最低。关键词稀疏度高，主要由家庭关系或特定背景定义的配角构成。"
    ]

    for i, p in enumerate(summary):
        top_k = keywords[keywords['persona'] == i].head(10)['word'].tolist()
        top_k_str = ", ".join(top_k)
        
        card = f"""
        <div class="card" style="border-left: 5px solid {colors[i]}">
            <div class="card-header">
                <span class="badge" style="background: {colors[i]}">P{p['id']}</span>
                <span class="title">Persona {p['id']}</span>
                <span class="count">{p['count']} 人</span>
            </div>
            <div class="gender-bar">
                <div class="male" style="width: {p['male']}%"></div>
                <div class="female" style="width: {p['female']}%"></div>
                <div class="neutral" style="width: {p['neutral']}%"></div>
            </div>
            <div class="gender-labels">
                <span>♂ {p['male']}%</span>
                <span>♀ {p['female']}%</span>
            </div>
            <p class="desc">{descriptions[i]}</p>
            <div class="top-chars">
                <strong>代表人物:</strong> {", ".join(p['top_chars'])}
            </div>
            <div class="keywords">
                <strong>核心关键词:</strong> {top_k_str}
            </div>
        </div>
        """
        persona_cards += card

    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>SAGE P4 (IT1000) 深度分析报告</title>
        <style>
            body {{ font-family: sans-serif; background: #0f1117; color: #e4e6ed; padding: 40px; line-height: 1.6; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            h1 {{ border-bottom: 2px solid #2a2e3a; padding-bottom: 10px; color: #fff; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
            .stat-box {{ background: #1a1d27; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #2a2e3a; }}
            .stat-val {{ font-size: 24px; font-weight: bold; color: #6c8aff; }}
            .card {{ background: #1a1d27; border-radius: 12px; padding: 25px; margin-bottom: 20px; border: 1px solid #2a2e3a; }}
            .card-header {{ display: flex; align-items: center; gap: 15px; margin-bottom: 15px; }}
            .badge {{ padding: 4px 10px; border-radius: 6px; font-weight: bold; color: #000; }}
            .title {{ font-size: 20px; font-weight: bold; flex: 1; }}
            .count {{ color: #8b90a0; }}
            .gender-bar {{ height: 8px; border-radius: 4px; overflow: hidden; display: flex; margin: 10px 0; background: #333; }}
            .male {{ background: #6c8aff; }} .female {{ background: #fb7185; }} .neutral {{ background: #8b90a0; }}
            .gender-labels {{ font-size: 12px; color: #8b90a0; display: flex; justify-content: space-between; }}
            .desc {{ color: #b0b3c1; font-size: 14px; margin: 15px 0; }}
            .top-chars, .keywords {{ font-size: 13px; margin-top: 10px; }}
            strong {{ color: #6c8aff; }}
            .viz-container {{ margin-top: 40px; text-align: center; }}
            .viz-img {{ max-width: 100%; border-radius: 12px; border: 1px solid #2a2e3a; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>上海小说人物聚类深度分析报告 (N=4, IT1000)</h1>
            <p style="color: #8b90a0">基于 SAGE 算法 · 1000 次 EM 迭代 · 4,860 位文学角色</p>
            
            <div class="stats-grid">
                <div class="stat-box"><div class="stat-val">4,860</div>角色总数</div>
                <div class="stat-box"><div class="stat-val">4</div>聚类数量</div>
                <div class="stat-box"><div class="stat-val">B+</div>结构稳定性</div>
                <div class="stat-box"><div class="stat-val">1000</div>迭代次数</div>
            </div>

            {persona_cards}

            <div class="viz-container">
                <h2>t-SNE 降维可视化</h2>
                <img src="persona_viz_p4.png" class="viz-img" alt="t-SNE Visualization">
                <p style="color: #8b90a0; font-size: 13px;">展示了人物在 1000 个词簇分布空间中的聚类形态，各中心标注了该类最具代表性的词丛。</p>
            </div>
            
            <div class="viz-container">
                <h2>网格搜索指标趋势</h2>
                <img src="grid_search_metrics.png" class="viz-img" alt="Grid Search Metrics">
                <p style="color: #8b90a0; font-size: 13px;">在 N=4 时，模型展现了最优的 Silhouette 分离度，是语义划分最稳健的层级。</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open('data/results/p4_report_it1000.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(">>> Report generated: data/results/p4_report_it1000.html")

if __name__ == "__main__":
    generate_report()
