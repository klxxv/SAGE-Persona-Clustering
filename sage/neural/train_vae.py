import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vae_model import NeuralSAGE, vae_loss
from tqdm import tqdm
import json
import time

def train_neural_sage(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Training Neural Variational SAGE on {device}")

    # 1. 加载数据
    print(">>> Loading and Preprocessing data...")
    df = pd.read_csv(args.data_file)
    df_words = pd.read_csv(args.word_csv_file)
    
    # 建立词表索引映射
    vocab_clusters = sorted(df_words['cluster_id'].unique())
    V = len(vocab_clusters)
    cluster_map = {c: i for i, c in enumerate(vocab_clusters)}
    
    # 提取 4 种角色角色
    roles = ['agent', 'patient', 'possessive', 'predicative']
    r_map = {r: i for i, r in enumerate(roles)}
    df = df[df['role'].isin(roles)].copy()
    
    # 角色映射
    char_keys = sorted((df['book'] + "_" + df['char_id'].astype(str)).unique())
    C = len(char_keys)
    char_map = {ck: i for i, ck in enumerate(char_keys)}
    
    df['c_idx'] = (df['book'] + "_" + df['char_id'].astype(str)).map(char_map)
    df['r_idx'] = df['role'].map(r_map)
    df['w_idx'] = df['cluster_id'].map(cluster_map)
    
    # 2. 构建 BoW 矩阵 (C x V)
    print(f">>> Building BoW matrix for {C} characters...")
    bow_matrix = torch.zeros((C, V))
    role_labels = torch.zeros(C, dtype=torch.long)
    
    # 聚合每个角色的总词频
    char_word_counts = df.groupby(['c_idx', 'w_idx'])['count'].sum().reset_index()
    bow_matrix[char_word_counts['c_idx'].values, char_word_counts['w_idx'].values] = torch.tensor(char_word_counts['count'].values, dtype=torch.float32)
    
    # 为每个角色分配主要角色 (用于 tau_r 的估计，这里取众数)
    char_main_role = df.groupby('c_idx')['r_idx'].agg(lambda x: x.value_counts().index[0]).reindex(range(C), fill_value=0)
    role_labels = torch.tensor(char_main_role.values, dtype=torch.long)

    # 3. 计算背景词频 (Baseline)
    total_counts = bow_matrix.sum(dim=0)
    bg_log_freq = torch.log(total_counts / total_counts.sum() + 1e-10)

    # 4. 初始化模型
    model = NeuralSAGE(vocab_size=V, n_personas=args.n_personas, n_roles=len(roles)).to(device)
    model.bg_log_freq.copy_(bg_log_freq.to(device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_loss = float('inf')
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 尝试加载最新断点
    if args.resume:
        latest_ckpt = os.path.join(ckpt_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_ckpt):
            print(f">>> Resuming from {latest_ckpt}...")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))

    dataset = TensorDataset(bow_matrix, role_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 5. 训练循环
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_recon = 0
        total_kl = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_bow, batch_role in pbar:
            batch_bow = batch_bow.to(device)
            batch_role = batch_role.to(device)

            optimizer.zero_grad()
            log_recon, mu, logvar, theta = model(batch_bow, batch_role)
            
            # 使用新的强化稀疏性损失函数
            loss, recon_loss, kl_loss, entropy, l1_loss = vae_loss(
                log_recon, batch_bow, mu, logvar, theta, model.eta_p,
                l1_lambda=args.l1_lambda, 
                entropy_lambda=args.entropy_lambda
            )
            
            # KL Annealing
            kl_weight = min(1.0, epoch / (args.epochs * 0.5))
            final_loss = recon_loss + kl_weight * kl_loss + (args.entropy_lambda * entropy) + (args.l1_lambda * l1_loss)

            final_loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            pbar.set_postfix({
                "Recon": f"{recon_loss:.1f}", 
                "KL": f"{kl_loss:.1f}",
                "Ent": f"{entropy:.2f}"
            })

        # 定期保存 Checkpoint
        avg_loss = (total_recon + total_kl) / len(loader)
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_ep{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss
            }, ckpt_path)

            # 同时保存一个 latest 指针
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, os.path.join(ckpt_dir, "latest_checkpoint.pt"))

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "vae_best_model.pt"))
            print(f"  *** New Best Model Saved (Loss: {best_loss:.4f})")

    # 6. 保存最终结果
    print(">>> Saving Neural VAE results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有角色的 Persona 分布 (Inference)
    model.eval()
    with torch.no_grad():
        all_h = model.encoder(bow_matrix.to(device))
        all_mu = model.fc_mu(all_h)
        all_theta = torch.softmax(all_mu, dim=-1).cpu().numpy()
        
    df_res = pd.DataFrame(all_theta, columns=[f"P{i}" for i in range(args.n_personas)])
    df_res['char_key'] = char_keys
    df_res['persona_label'] = np.argmax(all_theta, axis=1)
    
    df_res.to_csv(os.path.join(args.output_dir, "vae_persona_assignments.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "vae_model.pt"))
    
    # 提取 Persona 关键词 (从 eta_p 中提取)
    persona_keywords = {}
    eta_np = model.eta_p.detach().cpu().numpy()
    for p in range(args.n_personas):
        top_indices = np.argsort(eta_np[p])[-20:][::-1]
        persona_keywords[f"P{p}"] = [vocab_clusters[i] for i in top_indices]
        
    with open(os.path.join(args.output_dir, "vae_keywords.json"), 'w') as f:
        json.dump(persona_keywords, f, indent=4)
    
    print(f">>> Training Complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_csv_file', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="../../results/vae_results")
    parser.add_argument('--n_personas', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l1_lambda', type=float, default=1e-4, help="L1 regularization for eta_p")
    parser.add_argument('--entropy_lambda', type=float, default=1e-3, help="Entropy penalty for persona assignments")
    parser.add_argument('--resume', action='store_true', help="Resume from latest checkpoint")
    parser.add_argument('--save_interval', type=int, default=10, help="Epoch interval to save checkpoint")
    
    args = parser.parse_args()
    train_neural_sage(args)
