import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from src.quickdraw.dataset import load_quickdraw_experiment, misclassified_imgs, true_labels 
from src.quickdraw.models import QuickDrawCNN as Net
from src.quickdraw.dataset import cf_imgs as raw_cf_imgs
from src.quickdraw.dataset import true_labels as raw_true_labels, QD_CLASS_NAMES


import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def run_evaluation(run_dir, device):
    print("Loading pre-trained keras model...")
    keras_model_path = 'pretrained_models/cnn_quickdraw_5_classes.h5'
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    # Shape for keras: (30, 28, 28, 1)
    k_inputs = misclassified_imgs.astype('float32') 
    k_probs = keras_model.predict(k_inputs, verbose=0)

    o_mean_probs_keras = torch.tensor(k_probs)
    o_std_probs_keras = torch.zeros_like(o_mean_probs_keras) 
    o_final_preds_keras = torch.argmax(o_mean_probs_keras, dim=1)

    # Get group ID
    match = re.search(r'group(\d+)', os.path.basename(run_dir))
    group_id = int(match.group(1)) if match else -1
    is_cf_exp = "with_cf" in os.path.basename(run_dir)
    
    _, _, _, full_cf_loader, user_indices = load_quickdraw_experiment(
        group_id=group_id, 
        include_cf_in_train=is_cf_exp
    )

    # ensemble evaluation
    model_files = [f for f in os.listdir(run_dir) if f.endswith('.pth')]
    print(f"Path: {run_dir}, found {len(model_files)} model for ensemble evaluation...")

    all_cf_probs = []
    for model_file in model_files:
        model = Net().to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, model_file), map_location=device))
        model.eval()
        
        seed_cf_probs = []
        with torch.no_grad():
            for inputs, _ in full_cf_loader:
                outputs = model(inputs.to(device))
                seed_cf_probs.append(torch.softmax(outputs, dim=1).cpu())
        
        all_cf_probs.append(torch.cat(seed_cf_probs, dim=0))

    # --- Calculate stats ---
    stacked_cf_probs = torch.stack(all_cf_probs) # (Seeds, 150, 5)
    mean_probs = stacked_cf_probs.mean(dim=0)    # (150, 5)
    std_probs = stacked_cf_probs.std(dim=0)      # (150, 5)
    final_preds = torch.argmax(mean_probs, dim=1)

    # Calculate accuracy (training samples are masked out)
    num_samples, num_users = 30, 5
    mask = np.ones((num_users, num_samples), dtype=bool)
    if is_cf_exp:
        for s_idx, u_idx in enumerate(user_indices):
            mask[u_idx, s_idx] = False 
    
    cf_targets = full_cf_loader.dataset.targets
    flat_mask = mask.flatten()

    remaining_final_preds = final_preds[flat_mask].numpy()
    remaining_true_labels = cf_targets[flat_mask].numpy()

    total_accuracy = (remaining_final_preds == remaining_true_labels).mean()
    print(f"\nOverall Accuracy (Excl. Trained): {total_accuracy*100:.2f}%")

    # correct = (final_preds[flat_mask] == cf_targets[flat_mask])
    # print(f"✅ Ensemble accuracy (training samples excluded): {correct.float().mean()*100:.2f}%")

        # Evaluate Keras model on CFs
    print("Evaluating Baseline Keras model on Human CFs...")
    all_k_cf_probs = []
    for batch_imgs, _ in full_cf_loader:
        k_batch = batch_imgs.numpy().transpose(0, 2, 3, 1) if batch_imgs.shape[1] == 1 else batch_imgs.numpy()
        all_k_cf_probs.append(keras_model.predict(k_batch, verbose=0))
    
    k_cf_probs = np.concatenate(all_k_cf_probs, axis=0) # (150, 5)
    k_cf_preds = np.argmax(k_cf_probs, axis=1)        # (150,)

    k_remaining_preds = k_cf_preds[flat_mask]
    
    original_class_accs = []
    for label in range(5):
        label_mask = (remaining_true_labels == label)
        if np.any(label_mask):
            acc = (k_remaining_preds[label_mask] == label).mean()
            original_class_accs.append(acc)
        else:
            original_class_accs.append(0.0)
    original_avg_acc = (k_remaining_preds == remaining_true_labels).mean()

    class_stats = []
    all_seed_preds = torch.argmax(stacked_cf_probs, dim=2) # (Seeds, 150)
    
    for label in range(5):
        label_mask = (remaining_true_labels == label)
        if np.any(label_mask):
            ensemble_acc = (remaining_final_preds[label_mask] == label).mean()
            seed_accs = []
            for m_idx in range(stacked_cf_probs.shape[0]):
                m_preds = all_seed_preds[m_idx][flat_mask].numpy()
                seed_accs.append((m_preds[label_mask] == label).mean())
            seed_mean, seed_std = np.mean(seed_accs), np.std(seed_accs)
            count = np.sum(label_mask)
        else:
            ensemble_acc, seed_mean, seed_std, count = 0.0, 0.0, 0.0, 0
        
        class_stats.append({
            'Class': label,
            'ClassName': QD_CLASS_NAMES[label],
            'Ensemble_Accuracy': ensemble_acc,
            'Seed_Mean_Acc': seed_mean,
            'Seed_Std_Acc': seed_std,
            'Sample_Count': count
        })

    # Add Average
    class_stats.append({
        'Class': 'Average', 'ClassName': 'Average',
        'Ensemble_Accuracy': total_accuracy,
        'Seed_Mean_Acc': np.mean([(remaining_final_preds == remaining_true_labels).mean()]), # 简化
        'Seed_Std_Acc': 0.0, 'Sample_Count': len(remaining_true_labels)
    })

    df_stats = pd.DataFrame(class_stats)
    save_folder = os.path.join(run_dir, "results_qd")
    os.makedirs(save_folder, exist_ok=True)
    df_stats.to_csv(os.path.join(save_folder, "class_accuracy_report.csv"), index=False)
    
    # Plot class-wise accuracies with error bars
    plot_class_accuracies_qd(df_stats, original_class_accs, original_avg_acc, save_folder)

    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix_qd(remaining_true_labels, remaining_final_preds, save_folder)

    # Plot per-sample comparisons
    temp_imgs = raw_cf_imgs.copy()
    
    # If shape is (5, 30, 28, 28, 1) or (5, 30, 28, 28)
    if temp_imgs.shape[0] == 5 and temp_imgs.shape[1] == 30:
        print("Shape (User, Sample) transforming to (Sample, User)...")
        # Transpose: (5, 30, ...) -> (30, 5, ...)
        temp_imgs = np.transpose(temp_imgs, (1, 0, 2, 3, 4)) if temp_imgs.ndim == 5 else np.transpose(temp_imgs, (1, 0, 2, 3))

    if temp_imgs.ndim == 5:
        cf_imgs_prepared = temp_imgs.squeeze(-1)
    else:
        cf_imgs_prepared = temp_imgs

    print(f"Final shape: {cf_imgs_prepared.shape}") # (30, 5, 28, 28)

    save_folder = os.path.join(run_dir, "results_qd")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    plot_all_qd(
        mean_probs=mean_probs, 
        std_probs=std_probs, 
        final_preds=final_preds, 
        cf_imgs=cf_imgs_prepared,        
        original_imgs=misclassified_imgs, 
        true_labels=raw_true_labels, 
        mask=mask, 
        o_mean_probs=o_mean_probs_keras, 
        o_std_probs=o_std_probs_keras, 
        o_final_preds=o_final_preds_keras, 
        run_dir=run_dir, 
        is_cf_exp=is_cf_exp
    )

def plot_class_accuracies_qd(df_stats, original_class_accs, original_avg_acc, save_folder):
    plot_df = df_stats[df_stats['Class'] != 'Average'].copy()
    ensemble_accs = plot_df['Ensemble_Accuracy'].values * 100
    original_accs = np.array(original_class_accs) * 100
    class_names = plot_df['ClassName'].values
    
    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(12, 7))
    
    rects1 = plt.bar(x - width/2, original_accs, width, label='Original Keras (Baseline)', 
                     color='lightgray', edgecolor='gray')
    rects2 = plt.bar(x + width/2, ensemble_accs, width, label='Ensemble Model (Trained)', 
                     color='skyblue', edgecolor='navy')

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., h + 0.5, f'{h:.1f}%', 
                     ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    # Lines for average accuracies
    plt.axhline(y=original_avg_acc * 100, color='gray', linestyle='--', alpha=0.8, 
                label=f'Original Avg: {original_avg_acc*100:.1f}%')
    
    ensemble_avg = df_stats[df_stats['Class'] == 'Average']['Ensemble_Accuracy'].values[0] * 100
    plt.axhline(y=ensemble_avg, color='red', linestyle='--', linewidth=2,
                label=f'Ensemble Avg: {ensemble_avg:.1f}%')

    plt.title('Performance Comparison on Human CF Test Set\n(Original Model vs. Trained Ensemble)', fontsize=14)
    plt.xticks(x, class_names)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 115)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "overall_class_accuracies_comparison.png"))
    plt.close()
    print(f"📊 Comparison plot saved to: overall_class_accuracies_comparison.png")

def plot_confusion_matrix_qd(true_labels, pred_labels, save_folder):
    class_names = [QD_CLASS_NAMES[i] for i in range(5)]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(5))
    
    # Normalize
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_perc = np.nan_to_num(cm_perc)

    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar=True, square=True)
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f"({cm_perc[i, j]*100:.1f}%)", 
                     ha='center', va='center', color='black', fontsize=10)

    plt.title('Confusion Matrix: QuickDraw Ensemble Performance\n(Excluding Trained Samples)', fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "ensemble_confusion_matrix.png"))
    plt.close()
    print(f"✅ Confusion matrix saved to: ensemble_confusion_matrix.png")

def plot_all_qd(mean_probs, std_probs, 
                final_preds, cf_imgs, 
                original_imgs, true_labels, 
                mask, o_mean_probs, 
                o_std_probs, o_final_preds, 
                run_dir, is_cf_exp):
    
    class_names = [QD_CLASS_NAMES[i] for i in range(5)]
    num_samples = 30
    num_users = 5
    classes = np.arange(5)
    
    mean_probs_3d = mean_probs.reshape(num_samples, num_users, 5).numpy()
    std_probs_3d = std_probs.reshape(num_samples, num_users, 5).numpy()
    final_preds_2d = final_preds.reshape(num_samples, num_users).numpy()

    for s_idx in range(num_samples):
        s_true_label = true_labels[s_idx]
        true_label_name = QD_CLASS_NAMES[s_true_label]
        
        # Calculate sample accuracy
        s_mask = mask[:, s_idx] # (5,)
        s_test_preds = final_preds_2d[s_idx, s_mask]
        s_acc = (s_test_preds == s_true_label).mean() if len(s_test_preds) > 0 else 0.0

        fig, axes = plt.subplots(2, 6, figsize=(22, 8))
        plt.suptitle(f"QD Sample {s_idx} (True Label: {true_label_name}) | CF Test Acc: {s_acc*100:.1f}%", fontsize=16)

        # --- Column 1：Original Misclassified Image ---
        axes[0, 0].imshow(original_imgs[s_idx].squeeze(), cmap='gray')
        o_pred = o_final_preds[s_idx].item()
        o_pred_name = QD_CLASS_NAMES[o_pred]
        axes[0, 0].set_title(f"Original\nPred: {o_pred_name}", color='blue', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].bar(classes, o_mean_probs[s_idx].numpy(), yerr=o_std_probs[s_idx].numpy(), color='lightgray')
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].set_xticks(classes)
        axes[1, 0].set_xticklabels(class_names, rotation=45, fontsize=8)
        axes[1, 0].set_title("Original Probs", fontsize=9)

        # --- Column 2-6：Human CFs ---

        for u_idx in range(num_users):
            col = u_idx + 1
            img = cf_imgs[s_idx, u_idx]
            pred = final_preds_2d[s_idx, u_idx]
            pred_name = QD_CLASS_NAMES[pred]
            probs = mean_probs_3d[s_idx, u_idx]
            stds = std_probs_3d[s_idx, u_idx]
            
            is_trained = is_cf_exp and (not mask[u_idx, s_idx])

            axes[0, col].imshow(img.squeeze(), cmap='gray')
            if is_trained:
                t_color, t_text = 'orange', f"User {u_idx}\n(TRAINED)"
                for spine in axes[0, col].spines.values():
                    spine.set_edgecolor('orange'); spine.set_linewidth(4); spine.set_visible(True)
            else:
                is_correct = (pred == s_true_label)
                t_color = 'green' if is_correct else 'red'
                t_text = f"User {u_idx}\nPred: {pred_name}"
            
            axes[0, col].set_title(t_text, color=t_color, fontsize=10)
            axes[0, col].axis('off')

            ax_bar = axes[1, col]
            bars = ax_bar.bar(classes, probs, yerr=stds, color='skyblue', alpha=0.8, capsize=2)
            bars[pred].set_color(t_color)

            ax_bar.set_ylim(0, 1.1)
            ax_bar.set_xticks(classes)
            ax_bar.set_xticklabels(class_names, rotation=45, fontsize=8)
            if u_idx > 0: ax_bar.set_yticklabels([])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(run_dir, "results_qd", f"sample_{s_idx}_comparison.png")
        plt.savefig(save_path)
        plt.close()

    print(f"✅ All 30 samples plotted successfully.")