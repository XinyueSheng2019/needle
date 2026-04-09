"""
Example: How to use precision-optimized prediction

Run this after training your model to find optimal thresholds.
"""

from tensorflow.keras import models
from needle_train.preprocessing import preprocessing_untouched
from needle_train.precision_optimized_predict import (
    evaluate_with_different_thresholds,
    calibrate_thresholds_for_precision,
    predict_with_confidence_threshold
)
from needle_train.training import focal_loss
from needle_train.transient_model import *
from config import RAW_LABEL_DICT
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configuration
model_folder = 'focal_loss_3_label'
model_name = 'slsn_1500_tde_1500_d64_lr0.0001_b64_e100_focal_loss_2.5_0.25_0.45'  # Adjust to your model
model_path = f'../models/{model_folder}/{model_name}/best_model_recall/'

version = 'up_2000_stratified'
valid_path = f'{version}/hosted_set/valid_0.hdf5'
test_path = f'{version}/hosted_set/untouched_0.hdf5'
scaling_data_path = '../info/global_scaling_data_hosted_new.json'

# Load model
print("Loading model...")
custom_objects = {
    'F1PerClassMetrics': F1PerClassMetrics,
    'CustomLearningRateSchedule': CustomLearningRateSchedule,
    'PrecisionPerClassMetrics': PrecisionPerClassMetrics,
    'RecallPerClassMetrics': RecallPerClassMetrics,
    'focal_loss_fixed': focal_loss()
}

model = models.load_model(model_path, custom_objects=custom_objects)

# Load validation data
print("\nLoading validation data...")
valid_imageset, valid_metaset, valid_labels, _ = preprocessing_untouched(
    valid_path, RAW_LABEL_DICT, model_path,
    normalize_method=1, scaling_data_path=scaling_data_path, has_host=True
)

print(f"Validation set: {len(valid_labels)} samples")
print(f"Class distribution: {np.bincount(valid_labels.flatten())}")

# ============================================================
# Step 1: Standard prediction (baseline)
# ============================================================
print("\n" + "="*80)
print("STEP 1: Standard Prediction (Baseline)")
print("="*80)

standard_preds_probs = model.predict({'image_input': valid_imageset, 'meta_input': valid_metaset})
standard_preds = np.argmax(standard_preds_probs, axis=1)

print("\nStandard Prediction Results:")
print(classification_report(valid_labels.flatten(), standard_preds,
                           target_names=['SN', 'SLSN-I', 'TDE'],
                           digits=3))

# ============================================================
# Step 2: Evaluate different threshold combinations
# ============================================================
print("\n" + "="*80)
print("STEP 2: Evaluating Different Threshold Combinations")
print("="*80)

evaluate_with_different_thresholds(model, valid_imageset, valid_metaset, valid_labels)

# ============================================================
# Step 3: Auto-calibrate thresholds
# ============================================================
print("\n" + "="*80)
print("STEP 3: Auto-Calibrating Thresholds for Target Precision")
print("="*80)

optimal_thresholds = calibrate_thresholds_for_precision(
    model,
    valid_imageset,
    valid_metaset,
    valid_labels,
    target_precision={'SLSN-I': 0.50, 'TDE': 0.60},  # Adjust as needed
    target_recall={'SLSN-I': 0.55, 'TDE': 0.65}
)

print(f"\nOptimal thresholds found: {optimal_thresholds}")

# Apply optimal thresholds
opt_preds, opt_probs = predict_with_confidence_threshold(
    model, valid_imageset, valid_metaset, optimal_thresholds
)

print("\nOptimized Prediction Results:")
print(classification_report(valid_labels.flatten(), opt_preds,
                           target_names=['SN', 'SLSN-I', 'TDE'],
                           digits=3))

# ============================================================
# Step 4: Manual threshold tuning (if needed)
# ============================================================
print("\n" + "="*80)
print("STEP 4: Manual Threshold Example (High Precision Mode)")
print("="*80)

manual_thresholds = {
    'SN': 0.3,
    'SLSN-I': 0.75,  # Very high threshold for maximum precision
    'TDE': 0.65
}

manual_preds, _ = predict_with_confidence_threshold(
    model, valid_imageset, valid_metaset, manual_thresholds
)

print(f"\nManual thresholds: {manual_thresholds}")
print(classification_report(valid_labels.flatten(), manual_preds,
                           target_names=['SN', 'SLSN-I', 'TDE'],
                           digits=3))

# ============================================================
# Step 5: Confusion matrix comparison
# ============================================================
print("\n" + "="*80)
print("STEP 5: Confusion Matrix Comparison")
print("="*80)

print("\nStandard Prediction:")
print(confusion_matrix(valid_labels.flatten(), standard_preds))

print("\nOptimized Prediction (Auto-calibrated):")
print(confusion_matrix(valid_labels.flatten(), opt_preds))

print("\nManual High-Precision Prediction:")
print(confusion_matrix(valid_labels.flatten(), manual_preds))

# ============================================================
# Step 6: Recommendation
# ============================================================
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Calculate F1 scores
from sklearn.metrics import f1_score

standard_f1 = f1_score(valid_labels.flatten(), standard_preds, average='macro')
opt_f1 = f1_score(valid_labels.flatten(), opt_preds, average='macro')
manual_f1 = f1_score(valid_labels.flatten(), manual_preds, average='macro')

print(f"\nMacro F1 Scores:")
print(f"  Standard:     {standard_f1:.3f}")
print(f"  Optimized:    {opt_f1:.3f}")
print(f"  Manual (HP):  {manual_f1:.3f}")

if opt_f1 > standard_f1:
    print(f"\n✅ RECOMMENDED: Use optimized thresholds")
    print(f"   Thresholds: {optimal_thresholds}")
    print(f"   Improvement: +{(opt_f1 - standard_f1)*100:.1f}% F1 score")
else:
    print(f"\n✅ RECOMMENDED: Standard prediction is already good")
    print(f"   Consider adjusting training parameters instead of post-processing")

print("\n" + "="*80)
print("To use these thresholds in production:")
print(f"  1. Save thresholds: np.save('optimal_thresholds.npy', optimal_thresholds)")
print(f"  2. Load in predict script: thresholds = np.load('optimal_thresholds.npy', allow_pickle=True).item()")
print(f"  3. Use: predict_with_confidence_threshold(model, images, meta, thresholds)")
print("="*80)

