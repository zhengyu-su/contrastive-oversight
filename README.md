# Improving Human Oversight of AI Systems With Expert Feedback Using Interactive, Contrastive Explanations

This repository contains the official implementation and experimental pipelines for the workshop paper: **"Improving Human Oversight of AI Systems With Expert Feedback Using Interactive, Contrastive Explanations"**.

Instead of treating explanations as passive post-hoc visualizations for non-experts, this framework leverages human-generated **contrastive/counterfactual examples** to diagnose systematic model faults and realign AI decision boundaries with human expectations through robust training strategies.

---

## 📌 Overview

Our project is divided into two core experimental evaluations:
1. **Experiment 1 (Human-Model Divergences):** Quantifying the alignment gap between Convolutional Neural Networks (CNNs) and human experts using the **Semi-Factual Rate (SFR)** metric across MNIST and Google QuickDraw datasets.
2. **Experiment 2 (Training Strategies to Fix Model Faults):** Mitigating identified faults by incorporating human feedback via Data Augmentation, Adversarial Training (AT), and Counterfactual Training (CT).

---

## 🛠️ Environment Setup

We recommend using Conda to install the required dependencies and exact library versions used in our experiments. 

To recreate the environment and activate it, run the following commands in your terminal:

```bash
# Clone this repository
git clone https://github.com/zhengyu-su/contrastive-oversight.git
cd contrastive-oversight

# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the newly created environment
conda activate contrastive-oversight
```

---

## 🚀 Step-by-Step Experimental Pipeline

### Step 1: Experiment 1: Human-Model Divergence Analysis
This analysis is reproducible within the Jupyter Notebook: `notebooks/model_faults.ipynb`.
We explicitly assess the alignment divergences between human semantic classification and the model's predictions. 

#### 📊 Key Metric: Semi-Factual Rate (SFR)
> 
> Technically, when a user manually modifies an image to reflect the correct class, but the underlying model persists with its original incorrect prediction, the model treats these human changes as *semi-factuals* (inputs change, but the outcome prediction remains static). 
> 
> $$SFR = \frac{\text{Number of incorrect instances despite human intervention}}{\text{Number of total human counterfactual inputs}}$$
> 

---

### Step 2: Running Experiment 2 (Robust Re-Training)
Train the **Corrected Model** by combining standard Data Augmentation, Adversarial Training (AT), and Counterfactual Training (CT).
`group_id` indicates which randomly selected samples are being added to the training set during Counterfactual Training.


```bash
# Train the Corrected Model
python run_qd.py --group_id 0 --with_cf --exp_folder './experiments/quickdraw/aug_adv_cf' --adv_train --exp_name 'QD_ADV_CF'

python run_mnist.py --adv_train --with_cf --group_id 0 --exp_folder './experiments/mnist/aug_adv_cf' --exp_name 'MNIST_AUG_CF_ADV'
```

### Step 3: Final Evaluation
Evaluate the re-trained models against a held-out testing set of human-generated counterfactuals to measure the accuracy improvement and reduction in SFR.
```bash
python eval_qd.py --run_dir './experiments/quickdraw/aug_adv_cf/idx_group0_with_cf_adv_train_20260409_0235'

python eval_mnist.py --run_dir './experiments/mnist/with_cf/2_idx_group0_cf_run_20260417_1453' \
 --cf_data './data/mnist_cf/user_explanations_g2.npy'
```