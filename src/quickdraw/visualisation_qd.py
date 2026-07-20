import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from src.quickdraw.dataset import load_quickdraw_experiment, misclassified_imgs
from src.quickdraw.models import QuickDrawCNN as Net
from src.quickdraw.dataset import cf_imgs as raw_cf_imgs
from src.quickdraw.dataset import true_labels as raw_true_labels, QD_CLASS_NAMES

def run_evaluation_old(run_dir, device):
    # 1. 基础配置
    match = re.search(r'group(\d+)', os.path.basename(run_dir))
    group_id = int(match.group(1)) if match else -1
    is_cf_exp = "with_cf" in os.path.basename(run_dir)
    
    keras_model = tf.keras.models.load_model('pretrained_models/cnn_quickdraw_5_classes.h5')
    model_files = [f for f in os.listdir(run_dir) if f.endswith('.pth')]

    # 2. 加载数据
    _, _, _, full_cf_loader, user_indices = load_quickdraw_experiment(
        group_id=group_id, include_cf_in_train=is_cf_exp
    )
    
    orig_imgs_raw = misclassified_imgs.astype('float32') # (30, 28, 28, 1)
    cf_targets = full_cf_loader.dataset.targets.numpy()  # (150,)

    # 3. Ensemble 模型推理 (CF + Original)
    print(f"--- Starting Inference with {len(model_files)} models ---")
    ens_cf_probs_all = []
    ens_orig_probs_all = []
    
    for m_file in model_files:
        model = Net().to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, m_file), map_location=device))
        model.eval()
        
        # [CF 推理] 用 Loader 最保险
        seed_cf_probs = []
        with torch.no_grad():
            for inputs, _ in full_cf_loader:
                outputs = model(inputs.to(device))
                seed_cf_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        ens_cf_probs_all.append(np.concatenate(seed_cf_probs, axis=0))

        # [Original 推理]
        with torch.no_grad():
            orig_t = torch.from_numpy(orig_imgs_raw.transpose(0, 3, 1, 2)).to(device)
            outputs_orig = model(orig_t)
            ens_orig_probs_all.append(torch.softmax(outputs_orig, dim=1).cpu().numpy())

    ens_cf_probs_all = np.stack(ens_cf_probs_all)    # (Seeds, 150, 5)
    ens_orig_probs_all = np.stack(ens_orig_list_all) if 'ens_orig_list_all' in locals() else np.stack(ens_orig_probs_all) 
    
    ens_cf_probs_mean = ens_cf_probs_all.mean(axis=0)
    ens_cf_probs_std = ens_cf_probs_all.std(axis=0)
    ens_orig_probs_mean = ens_orig_probs_all.mean(axis=0)
    ens_orig_probs_std = ens_orig_probs_all.std(axis=0)

    # 4. Keras 推理 (CF + Original)
    # [Original]
    k_orig_probs = keras_model.predict(orig_imgs_raw, verbose=0)
    # [CF] 按照 Loader 顺序获得 Keras 预测
    all_k_cf_probs = []
    for batch_imgs, _ in full_cf_loader:
        k_batch = batch_imgs.numpy().transpose(0, 2, 3, 1)
        all_k_cf_probs.append(keras_model.predict(k_batch, verbose=0))
    k_cf_probs = np.concatenate(all_k_cf_probs, axis=0)

    # 5. Mask 计算
    mask = np.ones((5, 30), dtype=bool)
    if is_cf_exp:
        for s_idx, u_idx in enumerate(user_indices):
            mask[u_idx, s_idx] = False 
    flat_mask = mask.flatten()

    # 6. 生成 CSV 报告 (像 MNIST 一样)
    stats = []
    for c in range(5):
        c_mask = (cf_targets[flat_mask] == c)
        if not np.any(c_mask): continue
        
        ens_acc = (np.argmax(ens_cf_probs_mean[flat_mask], axis=1)[c_mask] == c).mean()
        k_acc = (np.argmax(k_cf_probs[flat_mask], axis=1)[c_mask] == c).mean()
        seed_accs = [(np.argmax(ens_cf_probs_all[s][flat_mask], axis=1)[c_mask] == c).mean() for s in range(len(model_files))]
        
        stats.append({
            'Class': QD_CLASS_NAMES[c],
            'Corrected_Accuracy': ens_acc,
            'Original_Baseline_Accuracy': k_acc,
            'Seed_Std': np.std(seed_accs),
            'Seed_Mean': np.mean(seed_accs),
            'Sample_Count': np.sum(c_mask)
        })

    all_ens_preds = np.argmax(ens_cf_probs_mean[flat_mask], axis=1)
    all_k_preds = np.argmax(k_cf_probs[flat_mask], axis=1)
    
    stats.append({
        'Class': 'Average',
        'Corrected_Accuracy': (all_ens_preds == cf_targets[flat_mask]).mean(),
        'Original_Baseline_Accuracy': (all_k_preds == cf_targets[flat_mask]).mean(),
        'Seed_Std': np.std([(np.argmax(ens_cf_probs_all[s][flat_mask], axis=1) == cf_targets[flat_mask]).mean() for s in range(len(model_files))]),
        'Seed_Mean': np.mean([(np.argmax(ens_cf_probs_all[s][flat_mask], axis=1) == cf_targets[flat_mask]).mean() for s in range(len(model_files))]),
        'Sample_Count': len(c_mask)
    })

    save_folder = os.path.join(run_dir, "single_sample_analysis_corrected_sequence")
    os.makedirs(save_folder, exist_ok=True)
    pd.DataFrame(stats).to_csv(os.path.join(save_folder, "comprehensive_accuracy_report.csv"), index=False)
    print(f"✅ Statistics saved to: comprehensive_accuracy_report.csv")

    # 7. 调用绘图 (CF 图片准备)
    cf_imgs_raw_copy = raw_cf_imgs.copy()
    if cf_imgs_raw_copy.ndim == 5: cf_imgs_raw_copy = cf_imgs_raw_copy.squeeze(-1)
    cf_imgs_prepared = np.transpose(cf_imgs_raw_copy, (1, 0, 2, 3)) 

    # plot_all_qd_final(
    #     cf_imgs_prepared, orig_imgs_raw, raw_true_labels,
    #     k_cf_probs, k_orig_probs,
    #     ens_cf_probs_mean, ens_cf_probs_std,
    #     ens_orig_probs_mean, ens_orig_probs_std,
    #     mask, save_folder, is_cf_exp
    # )

    # 重要：直接把 keras_model 传进去，不要传 k_cf_probs 数组
    plot_all_qd(
        cf_imgs_prepared, 
        orig_imgs_raw, 
        raw_true_labels,
        keras_model,  # 传入模型对象
        ens_cf_probs_mean, 
        ens_cf_probs_std,
        ens_orig_probs_mean, 
        ens_orig_probs_std,
        mask, 
        save_folder, 
        is_cf_exp
    )

def plot_all_qd_final(cf_imgs, orig_imgs, true_labels, k_cf_probs, k_orig_probs,
                     ens_cf_mean, ens_cf_std, ens_orig_mean, ens_orig_std,
                     mask, save_folder, is_cf_exp):
    
    class_names = [QD_CLASS_NAMES[i] for i in range(5)]
    num_samples, num_users = 30, 5
    classes = np.arange(5)

    for s_idx in range(num_samples):
        true_label = true_labels[s_idx]
        
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 6, hspace=0.1, wspace=0.3)
        plt.suptitle(f"Sample {s_idx} | True Label: {QD_CLASS_NAMES[true_label]}\nTop: Image | Mid: Original Baseline | Bottom: Corrected Model)", fontsize=24, y=1.1)

        # --- 第一列：Original ---
        inner_orig = gs[0, 0].subgridspec(3, 1, hspace=0.1)
        # Image
        ax_o_img = fig.add_subplot(inner_orig[0])
        ax_o_img.imshow(orig_imgs[s_idx].squeeze(), cmap='gray')
        ax_o_img.set_title(f"Original\nPred: {QD_CLASS_NAMES[np.argmax(k_orig_probs[s_idx])]} ", color='blue', fontsize=22)
        ax_o_img.axis('off')
        # Keras (Row 2)
        ax_o_k = fig.add_subplot(inner_orig[1])
        ax_o_k.bar(classes, k_orig_probs[s_idx], color='lightgray')
        ax_o_k.set_ylim(0, 1.1); ax_o_k.set_xticks(classes); ax_o_k.set_xticklabels([])
        # ax_o_k.set_ylabel("Original Baseline", fontsize=12)
        ax_o_k.set_box_aspect(0.6)

        # Ensemble (Row 3)
        ax_o_e = fig.add_subplot(inner_orig[2])
        ax_o_e.bar(classes, ens_orig_mean[s_idx], yerr=ens_orig_std[s_idx], color='skyblue', capsize=4)
        ax_o_e.set_ylim(0, 1.1); ax_o_e.set_xticks(classes)
        ax_o_e.set_xticklabels(class_names, rotation=45, fontsize=14, ha='right', rotation_mode='anchor')
        # ax_o_e.set_ylabel("Corrected Model", fontsize=12)
        ax_o_e.set_box_aspect(0.6)

        # --- 2-6 列：Human CFs ---
        for u_idx in range(num_users):
            col = u_idx + 1
            # flat_idx = u_idx * 30 + s_idx
            flat_idx = s_idx * num_users + u_idx
            inner_gs = gs[0, col].subgridspec(3, 1, hspace=0.1)
            
            is_trained = is_cf_exp and (not mask[u_idx, s_idx])
            ens_pred = np.argmax(ens_cf_mean[flat_idx])
            t_color = 'orange' if is_trained else ('green' if (ens_pred == true_label) else 'red')
            
            # (A) Image
            ax_img = fig.add_subplot(inner_gs[0])
            ax_img.imshow(cf_imgs[s_idx, u_idx], cmap='gray')
            ax_img.set_title(f"User {u_idx}\nPred: {QD_CLASS_NAMES[ens_pred]}", color=t_color, fontsize=22)
            ax_img.axis('off')

            # (B) Keras (Row 2)
            ax_k = fig.add_subplot(inner_gs[1])
            ax_k.bar(classes, k_cf_probs[flat_idx], color='lightgray')
            ax_k.set_ylim(0, 1.1); ax_k.set_xticks(classes); ax_k.set_xticklabels([])
            ax_k.set_yticklabels([])
            ax_k.set_box_aspect(0.6)

            # (C) Ensemble (Row 3)
            ax_e = fig.add_subplot(inner_gs[2])
            e_bars = ax_e.bar(classes, ens_cf_mean[flat_idx], yerr=ens_cf_std[flat_idx], color='skyblue', capsize=4)
            e_bars[ens_pred].set_color(t_color)
            ax_e.set_ylim(0, 1.1); ax_e.set_xticks(classes)
            ax_e.set_xticklabels(class_names, rotation=45, fontsize=14, ha='right', rotation_mode='anchor')
            ax_e.set_yticklabels([])
            ax_e.set_box_aspect(0.6)

        plt.savefig(os.path.join(save_folder, f"sample_{s_idx}_comparison.png"), bbox_inches='tight', dpi=150)
        plt.close()

def  plot_all_qd(cf_imgs, orig_imgs, true_labels, keras_model, # 注意这里传入的是模型对象
                     ens_cf_mean, ens_cf_std, ens_orig_mean, ens_orig_std,
                     mask, save_folder, is_cf_exp):
    
    class_names = ['giraffe', 'mushroom', 'bicycle', 'helicopter', 'pizza']
    num_samples, num_users = 30, 5
    classes = np.arange(5)
    print(f"orig img min: {orig_imgs.min()}, max: {orig_imgs.max()} | CF img min: {cf_imgs.min()}, max: {cf_imgs.max()}")

    for s_idx in range(num_samples):
        true_label_idx = true_labels[s_idx]
        true_label_name = class_names[true_label_idx]
        
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 6, hspace=0.1, wspace=0.3)
        plt.suptitle(f"Sample {s_idx} | True Label: {true_label_name}\n"
                     f"Top: Image | Mid: Original Baseline | Bottom: Corrected Model", 
                     fontsize=24, y=1.1)

        # --- 第一列：Original ---
        inner_orig = gs[0, 0].subgridspec(3, 1, hspace=0.1)
        
        # (A) Original Image
        ax_o_img = fig.add_subplot(inner_orig[0])
        ax_o_img.imshow(orig_imgs[s_idx].squeeze(), cmap='gray')
        
        # 实时预测 Original
        k_orig_prob = keras_model.predict(orig_imgs[s_idx].reshape(1, 28, 28, 1), verbose=0).squeeze()
        k_orig_pred = np.argmax(k_orig_prob)
        
        ax_o_img.set_title(f"Original\nPred: {class_names[k_orig_pred]} ", 
                           color='blue', fontsize=22)
        ax_o_img.axis('off')

        # (B) Keras Baseline (Row 2)
        ax_o_k = fig.add_subplot(inner_orig[1])
        ax_o_k.bar(classes, k_orig_prob, color='lightgray')
        ax_o_k.set_ylim(0, 1.1); ax_o_k.set_xticks(classes); ax_o_k.set_xticklabels([])
        ax_o_k.set_box_aspect(0.6)

        # (C) Ensemble (Row 3)
        ax_o_e = fig.add_subplot(inner_orig[2])
        ax_o_e.bar(classes, ens_orig_mean[s_idx], yerr=ens_orig_std[s_idx], color='skyblue', capsize=4)
        ax_o_e.set_ylim(0, 1.1); ax_o_e.set_xticks(classes)
        ax_o_e.set_xticklabels(class_names, rotation=45, fontsize=14, ha='right', rotation_mode='anchor')
        ax_o_e.set_box_aspect(0.6)

        # --- 第 2-6 列：Human CFs ---
        for u_idx in range(num_users):
            col = u_idx + 1
            flat_idx = s_idx * num_users + u_idx # 仅用于索引 Ensemble 数据
            inner_gs = gs[0, col].subgridspec(3, 1, hspace=0.1)
            
            # (A) CF Image
            ax_img = fig.add_subplot(inner_gs[0])
            current_cf_img = cf_imgs[s_idx, u_idx]
            ax_img.imshow(current_cf_img.squeeze(), cmap='gray')
            
            # 实时预测当前 CF 图片
            k_cf_prob_single = keras_model.predict(current_cf_img.reshape(1, 28, 28, 1), verbose=0).squeeze()
            
            is_trained = is_cf_exp and (not mask[u_idx, s_idx])
            ens_pred = np.argmax(ens_cf_mean[flat_idx])
            t_color = 'orange' if is_trained else ('green' if (ens_pred == true_label_idx) else 'red')
            
            ax_img.set_title(f"User {u_idx}\nPred: {class_names[ens_pred]}", color=t_color, fontsize=22)
            ax_img.axis('off')

            # (B) Keras Baseline (Row 2) - 实时预测结果
            ax_k = fig.add_subplot(inner_gs[1])
            ax_k.bar(classes, k_cf_prob_single, color='lightgray')
            ax_k.set_ylim(0, 1.1); ax_k.set_xticks(classes); ax_k.set_xticklabels([])
            ax_k.set_yticklabels([]); ax_k.set_box_aspect(0.6)

            # (C) Ensemble (Row 3)
            ax_e = fig.add_subplot(inner_gs[2])
            e_bars = ax_e.bar(classes, ens_cf_mean[flat_idx], yerr=ens_cf_std[flat_idx], color='skyblue', capsize=4)
            e_bars[ens_pred].set_color(t_color)
            ax_e.set_ylim(0, 1.1); ax_e.set_xticks(classes)
            ax_e.set_xticklabels(class_names, rotation=45, fontsize=14, ha='right', rotation_mode='anchor')
            ax_e.set_yticklabels([]); ax_e.set_box_aspect(0.6)

        plt.savefig(os.path.join(save_folder, f"sample_{s_idx}_comparison.png"), bbox_inches='tight', dpi=150)
        plt.close()

def run_evaluation(run_dir, device):
    # 1. 基础配置
    match = re.search(r'group(\d+)', os.path.basename(run_dir))
    group_id = int(match.group(1)) if match else -1
    is_cf_exp = "with_cf" in os.path.basename(run_dir)
    
    keras_model = tf.keras.models.load_model('pretrained_models/cnn_quickdraw_5_classes.h5')
    model_files = [f for f in os.listdir(run_dir) if f.endswith('.pth')]

    # 2. 加载数据
    _, _, _, full_cf_loader, user_indices = load_quickdraw_experiment(
        group_id=group_id, include_cf_in_train=is_cf_exp
    )
    
    orig_imgs_raw = misclassified_imgs.astype('float32') # (30, 28, 28, 1)
    cf_targets = full_cf_loader.dataset.targets.numpy()  # (150,)

    # 3. Ensemble 模型推理 (CF + Original)
    print(f"--- Starting Inference with {len(model_files)} models ---")
    ens_cf_probs_all = []
    ens_orig_probs_all = []
    
    for m_file in model_files:
        model = Net().to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, m_file), map_location=device))
        model.eval()
        
        # [CF 推理] 用 Loader 最保险
        seed_cf_probs = []
        with torch.no_grad():
            for inputs, _ in full_cf_loader:
                outputs = model(inputs.to(device))
                seed_cf_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        ens_cf_probs_all.append(np.concatenate(seed_cf_probs, axis=0))

        # [Original 推理]
        with torch.no_grad():
            orig_t = torch.from_numpy(orig_imgs_raw.transpose(0, 3, 1, 2)).to(device)
            outputs_orig = model(orig_t)
            ens_orig_probs_all.append(torch.softmax(outputs_orig, dim=1).cpu().numpy())

    ens_cf_probs_all = np.stack(ens_cf_probs_all)    # (Seeds, 150, 5)
    ens_orig_probs_all = np.stack(ens_orig_list_all) if 'ens_orig_list_all' in locals() else np.stack(ens_orig_probs_all) 
    
    ens_cf_probs_mean = ens_cf_probs_all.mean(axis=0)
    ens_cf_probs_std = ens_cf_probs_all.std(axis=0)
    ens_orig_probs_mean = ens_orig_probs_all.mean(axis=0)
    ens_orig_probs_std = ens_orig_probs_all.std(axis=0)

    # 4. Keras 推理 (CF + Original)
    # [Original]
    k_orig_probs = keras_model.predict(orig_imgs_raw, verbose=0)
    # [CF] 按照 Loader 顺序获得 Keras 预测
    all_k_cf_probs = []
    for batch_imgs, _ in full_cf_loader:

        k_batch = batch_imgs.numpy().transpose(0, 2, 3, 1)
        k_batch = k_batch + 0.5
        print(f"batch image range:{k_batch.min()} to {k_batch.max()}")  # Debug: 检查输入范围

        all_k_cf_probs.append(keras_model.predict(k_batch, verbose=0))
    k_cf_probs = np.concatenate(all_k_cf_probs, axis=0)

    # 5. Mask 计算
    mask = np.ones((5, 30), dtype=bool)
    if is_cf_exp:
        for s_idx, u_idx in enumerate(user_indices):
            mask[u_idx, s_idx] = False 
    flat_mask = mask.T.flatten()


    # 6. 生成 CSV 报告 (像 MNIST 一样)
    stats = []
    for c in range(5):
        c_mask = (cf_targets[flat_mask] == c)
        # breakpoint()  # Debug: 检查 c_mask 的正确性
        if not np.any(c_mask): continue
        
        ens_acc = (np.argmax(ens_cf_probs_mean[flat_mask], axis=1)[c_mask] == c).mean()
        k_acc = (np.argmax(k_cf_probs[flat_mask], axis=1)[c_mask] == c).mean()
        seed_accs = [(np.argmax(ens_cf_probs_all[s][flat_mask], axis=1)[c_mask] == c).mean() for s in range(len(model_files))]
        
        stats.append({
            'Class': QD_CLASS_NAMES[c],
            'Corrected_Accuracy': ens_acc,
            'Original_Baseline_Accuracy': k_acc,
            'Seed_Std': np.std(seed_accs),
            'Seed_Mean': np.mean(seed_accs),
            'Sample_Count': np.sum(c_mask)
        })

    all_ens_preds = np.argmax(ens_cf_probs_mean[flat_mask], axis=1)
    all_k_preds = np.argmax(k_cf_probs[flat_mask], axis=1)
    
    stats.append({
        'Class': 'Average',
        'Corrected_Accuracy': (all_ens_preds == cf_targets[flat_mask]).mean(),
        'Original_Baseline_Accuracy': (all_k_preds == cf_targets[flat_mask]).mean(),
        'Seed_Std': np.std([(np.argmax(ens_cf_probs_all[s][flat_mask], axis=1) == cf_targets[flat_mask]).mean() for s in range(len(model_files))]),
        'Seed_Mean': np.mean([(np.argmax(ens_cf_probs_all[s][flat_mask], axis=1) == cf_targets[flat_mask]).mean() for s in range(len(model_files))]),
        'Sample_Count': len(c_mask)
    })

    # breakpoint()  # Debug: 检查 stats 的正确性

    save_folder = os.path.join(run_dir, "single_sample_analysis_corrected_sequence")
    os.makedirs(save_folder, exist_ok=True)
    pd.DataFrame(stats).to_csv(os.path.join(save_folder, "comprehensive_accuracy_report.csv"), index=False)
    print(f"✅ Statistics saved to: comprehensive_accuracy_report.csv")

    # 7. 调用绘图 (CF 图片准备)
    cf_imgs_raw_copy = raw_cf_imgs.copy()
    if cf_imgs_raw_copy.ndim == 5: cf_imgs_raw_copy = cf_imgs_raw_copy.squeeze(-1)
    cf_imgs_prepared = np.transpose(cf_imgs_raw_copy, (1, 0, 2, 3)) 

    # plot_all_qd_final(
    #     cf_imgs_prepared, orig_imgs_raw, raw_true_labels,
    #     k_cf_probs, k_orig_probs,
    #     ens_cf_probs_mean, ens_cf_probs_std,
    #     ens_orig_probs_mean, ens_orig_probs_std,
    #     mask, save_folder, is_cf_exp
    # )

    # 重要：直接把 keras_model 传进去，不要传 k_cf_probs 数组
    plot_all_qd(
        cf_imgs_prepared, 
        orig_imgs_raw, 
        raw_true_labels,
        keras_model,  # 传入模型对象
        ens_cf_probs_mean, 
        ens_cf_probs_std,
        ens_orig_probs_mean, 
        ens_orig_probs_std,
        mask, 
        save_folder, 
        is_cf_exp
    )