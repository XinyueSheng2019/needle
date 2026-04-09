"""
Evaluate Model After Training

Use this script to compute all metrics after training completes.
This replaces the monitoring callbacks that were disabled for M1 stability.

Usage:
    python evaluate_model_post_training.py

Make sure to update MODEL_PATH, VERSION below to match your trained model.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU mode

from tensorflow.keras import models
from needle_train.preprocessing import preprocessing_untouched
from needle_train.training import focal_loss
from needle_train.transient_model import *
from config import RAW_LABEL_DICT
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_fscore_support
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================

MODEL_FOLDER = 'no_focal_loss'
MODEL_NAME = 'slsn_1500_tde_1500_d64_lr0.0001_b32_e100_no_weight'  # Update to your model name
MODEL_PATH = f'../models/{MODEL_FOLDER}/{MODEL_NAME}/best_model'

VERSION = 'up_2000_stratified'
VALID_PATH = f'{VERSION}/hosted_set/valid_0.hdf5'
TEST_PATH = f'{VERSION}/hosted_set/untouched_0.hdf5'
SCALING_DATA_PATH = '../info/global_scaling_data_hosted_new.json'

OUTPUT_DIR = f'../models/{MODEL_FOLDER}/{MODEL_NAME}/post_training_eval'

# ============================================================
# LOAD MODEL
# ============================================================

print("="*80)
print("LOADING MODEL")
print("="*80)
print(f"Model path: {MODEL_PATH}")

custom_objects = {
    'F1PerClassMetrics': F1PerClassMetrics,
    'CustomLearningRateSchedule': CustomLearningRateSchedule,
    'PrecisionPerClassMetrics': PrecisionPerClassMetrics,
    'RecallPerClassMetrics': RecallPerClassMetrics,
    'focal_loss_fixed': focal_loss()
}

try:
    model = models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nTrying with basic custom objects...")
    model = models.load_model(MODEL_PATH, custom_objects={'F1PerClassMetrics': F1PerClassMetrics})
    print("✓ Model loaded with basic custom objects")

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

print("Loading validation set...")
valid_images, valid_meta, valid_labels, _ = preprocessing_untouched(
    VALID_PATH, RAW_LABEL_DICT, MODEL_PATH,
    normalize_method=1, scaling_data_path=SCALING_DATA_PATH, has_host=True
)
print(f"✓ Validation: {len(valid_labels)} samples")
print(f"  Class distribution: {np.bincount(valid_labels.flatten())}")

print("\nLoading test set...")
test_images, test_meta, test_labels, _ = preprocessing_untouched(
    TEST_PATH, RAW_LABEL_DICT, MODEL_PATH,
    normalize_method=1, scaling_data_path=SCALING_DATA_PATH, has_host=True
)
print(f"✓ Test: {len(test_labels)} samples")
print(f"  Class distribution: {np.bincount(test_labels.flatten())}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# VALIDATION SET EVALUATION
# ============================================================

print("\n" + "="*80)
print("VALIDATION SET EVALUATION")
print("="*80)

val_preds_probs = model.predict({'image_input': valid_images, 'meta_input': valid_meta})
val_preds = np.argmax(val_preds_probs, axis=1)

print("\nClassification Report:")
print(classification_report(valid_labels.flatten(), val_preds,
                           target_names=['SN', 'SLSN-I', 'TDE'],
                           digits=3))

# Confusion matrix
cm = confusion_matrix(valid_labels.flatten(), val_preds)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['SN', 'SLSN-I', 'TDE'],
           yticklabels=['SN', 'SLSN-I', 'TDE'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Validation Set Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/val_confusion_matrix.png', dpi=150)
print(f"✓ Saved: {OUTPUT_DIR}/val_confusion_matrix.png")

# SLSN-I specific metrics
print("\n" + "-"*80)
print("SLSN-I Detailed Analysis")
print("-"*80)

slsn_mask = valid_labels.flatten() == 1
if np.sum(slsn_mask) > 0:
    slsn_correct = np.sum(val_preds[slsn_mask] == 1)
    slsn_total = np.sum(slsn_mask)
    predicted_as_slsn = np.sum(val_preds == 1)
    
    slsn_recall = slsn_correct / slsn_total
    slsn_precision = slsn_correct / predicted_as_slsn if predicted_as_slsn > 0 else 0
    slsn_f1 = 2 * slsn_precision * slsn_recall / (slsn_precision + slsn_recall) if (slsn_precision + slsn_recall) > 0 else 0
    
    print(f"Recall:    {slsn_recall:.3f} ({slsn_correct}/{slsn_total} correctly identified)")
    print(f"Precision: {slsn_precision:.3f} ({slsn_correct}/{predicted_as_slsn} true positives)")
    print(f"F1 Score:  {slsn_f1:.3f}")
    
    # Misclassification analysis
    slsn_preds = val_preds[slsn_mask]
    as_sn = np.sum(slsn_preds == 0)
    as_tde = np.sum(slsn_preds == 2)
    print(f"\nMisclassified SLSN-I:")
    print(f"  As SN:  {as_sn}/{slsn_total} ({as_sn/slsn_total*100:.1f}%)")
    print(f"  As TDE: {as_tde}/{slsn_total} ({as_tde/slsn_total*100:.1f}%)")

# TDE specific metrics
print("\n" + "-"*80)
print("TDE Detailed Analysis")
print("-"*80)

tde_mask = valid_labels.flatten() == 2
if np.sum(tde_mask) > 0:
    tde_correct = np.sum(val_preds[tde_mask] == 2)
    tde_total = np.sum(tde_mask)
    predicted_as_tde = np.sum(val_preds == 2)
    
    tde_recall = tde_correct / tde_total
    tde_precision = tde_correct / predicted_as_tde if predicted_as_tde > 0 else 0
    tde_f1 = 2 * tde_precision * tde_recall / (tde_precision + tde_recall) if (tde_precision + tde_recall) > 0 else 0
    
    print(f"Recall:    {tde_recall:.3f} ({tde_correct}/{tde_total} correctly identified)")
    print(f"Precision: {tde_precision:.3f} ({tde_correct}/{predicted_as_tde} true positives)")
    print(f"F1 Score:  {tde_f1:.3f}")
    
    # Misclassification analysis
    tde_preds = val_preds[tde_mask]
    as_sn = np.sum(tde_preds == 0)
    as_slsn = np.sum(tde_preds == 1)
    print(f"\nMisclassified TDE:")
    print(f"  As SN:    {as_sn}/{tde_total} ({as_sn/tde_total*100:.1f}%)")
    print(f"  As SLSN-I: {as_slsn}/{tde_total} ({as_slsn/tde_total*100:.1f}%)")

# ROC AUC
try:
    roc_auc = roc_auc_score(valid_labels.flatten(), val_preds_probs,
                            multi_class='ovr', average='macro')
    print(f"\nROC AUC (macro): {roc_auc:.3f}")
except Exception as e:
    print(f"\nROC AUC calculation failed: {e}")

# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\n" + "="*80)
print("TEST SET (UNTOUCHED) EVALUATION")
print("="*80)

test_preds_probs = model.predict({'image_input': test_images, 'meta_input': test_meta})
test_preds = np.argmax(test_preds_probs, axis=1)

print("\nClassification Report:")
print(classification_report(test_labels.flatten(), test_preds,
                           target_names=['SN', 'SLSN-I', 'TDE'],
                           digits=3))

# Confusion matrix
cm_test = confusion_matrix(test_labels.flatten(), test_preds)
print("\nConfusion Matrix:")
print(cm_test)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
           xticklabels=['SN', 'SLSN-I', 'TDE'],
           yticklabels=['SN', 'SLSN-I', 'TDE'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Set Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/test_confusion_matrix.png', dpi=150)
print(f"✓ Saved: {OUTPUT_DIR}/test_confusion_matrix.png")

# Test set SLSN-I metrics
print("\n" + "-"*80)
print("Test Set - SLSN-I Performance")
print("-"*80)

test_slsn_mask = test_labels.flatten() == 1
if np.sum(test_slsn_mask) > 0:
    test_slsn_correct = np.sum(test_preds[test_slsn_mask] == 1)
    test_slsn_total = np.sum(test_slsn_mask)
    test_predicted_as_slsn = np.sum(test_preds == 1)
    
    test_slsn_recall = test_slsn_correct / test_slsn_total
    test_slsn_precision = test_slsn_correct / test_predicted_as_slsn if test_predicted_as_slsn > 0 else 0
    test_slsn_f1 = 2 * test_slsn_precision * test_slsn_recall / (test_slsn_precision + test_slsn_recall) if (test_slsn_precision + test_slsn_recall) > 0 else 0
    
    print(f"Recall:    {test_slsn_recall:.3f} ({test_slsn_correct}/{test_slsn_total})")
    print(f"Precision: {test_slsn_precision:.3f} ({test_slsn_correct}/{test_predicted_as_slsn})")
    print(f"F1 Score:  {test_slsn_f1:.3f}")

# Test set TDE metrics
print("\n" + "-"*80)
print("Test Set - TDE Performance")
print("-"*80)

test_tde_mask = test_labels.flatten() == 2
if np.sum(test_tde_mask) > 0:
    test_tde_correct = np.sum(test_preds[test_tde_mask] == 2)
    test_tde_total = np.sum(test_tde_mask)
    test_predicted_as_tde = np.sum(test_preds == 2)
    
    test_tde_recall = test_tde_correct / test_tde_total
    test_tde_precision = test_tde_correct / test_predicted_as_tde if test_predicted_as_tde > 0 else 0
    test_tde_f1 = 2 * test_tde_precision * test_tde_recall / (test_tde_precision + test_tde_recall) if (test_tde_precision + test_tde_recall) > 0 else 0
    
    print(f"Recall:    {test_tde_recall:.3f} ({test_tde_correct}/{test_tde_total})")
    print(f"Precision: {test_tde_precision:.3f} ({test_tde_correct}/{test_predicted_as_tde})")
    print(f"F1 Score:  {test_tde_f1:.3f}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

summary = f"""
Model: {MODEL_NAME}

VALIDATION SET:
  SLSN-I: Precision={slsn_precision:.3f}, Recall={slsn_recall:.3f}, F1={slsn_f1:.3f}
  TDE:    Precision={tde_precision:.3f}, Recall={tde_recall:.3f}, F1={tde_f1:.3f}

TEST SET:
  SLSN-I: Precision={test_slsn_precision:.3f}, Recall={test_slsn_recall:.3f}, F1={test_slsn_f1:.3f}
  TDE:    Precision={test_tde_precision:.3f}, Recall={test_tde_recall:.3f}, F1={test_tde_f1:.3f}

Output directory: {OUTPUT_DIR}
"""

print(summary)

# Save summary to file
with open(f'{OUTPUT_DIR}/evaluation_summary.txt', 'w') as f:
    f.write(summary)
    f.write("\n\nVALIDATION SET Classification Report:\n")
    f.write(classification_report(valid_labels.flatten(), val_preds,
                                 target_names=['SN', 'SLSN-I', 'TDE'],
                                 digits=3))
    f.write("\n\nTEST SET Classification Report:\n")
    f.write(classification_report(test_labels.flatten(), test_preds,
                                 target_names=['SN', 'SLSN-I', 'TDE'],
                                 digits=3))

print(f"✓ Saved: {OUTPUT_DIR}/evaluation_summary.txt")
print("\nEvaluation complete!")

