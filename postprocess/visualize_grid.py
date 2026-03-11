import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_grid_search(json_file, output_img="grid_search_metrics.png"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(15, 6))
    
    # Plot Perplexity
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x="n_personas", y="perplexity", hue="em_iters", marker='o')
    plt.title("Perplexity vs. Persona Count")
    plt.grid(True)
    
    # Plot Silhouette
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x="n_personas", y="silhouette", hue="em_iters", marker='o')
    plt.title("Silhouette Score (EMD) vs. Persona Count")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_img)
    print(f">>> Grid search metrics plot saved to {output_img}")

if __name__ == "__main__":
    plot_grid_search("grid_search_results.json")
