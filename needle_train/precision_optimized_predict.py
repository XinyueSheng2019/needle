"""
Precision-Optimized Prediction Module

This module provides prediction strategies optimized for high precision,
reducing false positives for SLSN-I and TDE classifications.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve


def predict_with_confidence_threshold(model, images, meta, 
                                       thresholds={'SN': 0.3, 'SLSN-I': 0.6, 'TDE': 0.5},
                                       default_class=0):
    """
    Predict with class-specific confidence thresholds to improve precision.
    
    Args:
        model: Trained model
        images: Image input
        meta: Metadata input
        thresholds: Dict with confidence thresholds for each class
        default_class: Default class when no threshold is met (usually SN=0)
    
    Returns:
        predictions: Class predictions
        probabilities: Raw probabilities
    """
    # Get raw predictions
    probabilities = model.predict({'image_input': images, 'meta_input': meta})
    predictions = np.zeros(len(probabilities), dtype=int) + default_class
    
    # Apply thresholds
    for i in range(len(probabilities)):
        pred_class = np.argmax(probabilities[i])
        confidence = probabilities[i, pred_class]
        
        # For SLSN-I (class 1) - require high confidence
        if pred_class == 1 and confidence >= thresholds['SLSN-I']:
            predictions[i] = 1
        # For TDE (class 2) - require medium-high confidence  
        elif pred_class == 2 and confidence >= thresholds['TDE']:
            predictions[i] = 2
        # For SN (class 0) - require low confidence (default class)
        elif pred_class == 0 and confidence >= thresholds['SN']:
            predictions[i] = 0
        # If no threshold met, use default (SN)
        else:
            predictions[i] = default_class
            
    return predictions, probabilities


def calibrate_thresholds_for_precision(model, val_images, val_meta, val_labels,
                                       target_precision={'SLSN-I': 0.50, 'TDE': 0.60},
                                       target_recall={'SLSN-I': 0.60, 'TDE': 0.70}):
    """
    Automatically calibrate confidence thresholds to achieve target precision.
    
    Args:
        model: Trained model
        val_images, val_meta, val_labels: Validation data
        target_precision: Dict with target precision for each rare class
        target_recall: Dict with minimum acceptable recall
    
    Returns:
        optimal_thresholds: Dict with calibrated thresholds
    """
    # Get predictions
    probabilities = model.predict({'image_input': val_images, 'meta_input': val_meta})
    
    optimal_thresholds = {'SN': 0.3}  # SN is default, low threshold
    
    # Calibrate for SLSN-I (class 1)
    slsn_mask = val_labels.flatten() == 1
    if np.sum(slsn_mask) > 0:
        # Get probabilities for SLSN-I class
        slsn_probs = probabilities[:, 1]
        
        # Find threshold that gives target precision while maintaining recall
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.3, 0.95, 0.05):
            preds = (slsn_probs >= threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum(preds[slsn_mask] == 1)
            fp = np.sum(preds[~slsn_mask] == 1)
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
                recall = tp / np.sum(slsn_mask)
                
                if precision >= target_precision['SLSN-I'] and recall >= target_recall['SLSN-I']:
                    f1 = 2 * precision * recall / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
        
        optimal_thresholds['SLSN-I'] = best_threshold
        print(f"SLSN-I threshold: {best_threshold:.2f}")
    
    # Calibrate for TDE (class 2)
    tde_mask = val_labels.flatten() == 2
    if np.sum(tde_mask) > 0:
        tde_probs = probabilities[:, 2]
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.3, 0.95, 0.05):
            preds = (tde_probs >= threshold).astype(int)
            
            tp = np.sum(preds[tde_mask] == 1)
            fp = np.sum(preds[~tde_mask] == 1)
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
                recall = tp / np.sum(tde_mask)
                
                if precision >= target_precision['TDE'] and recall >= target_recall['TDE']:
                    f1 = 2 * precision * recall / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
        
        optimal_thresholds['TDE'] = best_threshold
        print(f"TDE threshold: {best_threshold:.2f}")
    
    return optimal_thresholds


def predict_with_relative_confidence(model, images, meta, 
                                     confidence_margins={'SLSN-I': 0.15, 'TDE': 0.10}):
    """
    Predict using relative confidence margins.
    Only classify as rare class if it has significantly higher probability than others.
    
    Args:
        model: Trained model
        images, meta: Input data
        confidence_margins: Minimum margin over second-best class
    
    Returns:
        predictions: Class predictions
    """
    probabilities = model.predict({'image_input': images, 'meta_input': meta})
    predictions = np.zeros(len(probabilities), dtype=int)
    
    for i in range(len(probabilities)):
        probs = probabilities[i]
        top_two = np.argsort(probs)[-2:][::-1]  # Two highest probabilities
        best_class = top_two[0]
        second_best = top_two[1]
        
        margin = probs[best_class] - probs[second_best]
        
        # For SLSN-I, require large margin
        if best_class == 1:
            if margin >= confidence_margins['SLSN-I']:
                predictions[i] = 1
            else:
                predictions[i] = 0  # Default to SN if not confident
        # For TDE, require medium margin
        elif best_class == 2:
            if margin >= confidence_margins['TDE']:
                predictions[i] = 2
            else:
                predictions[i] = 0
        # For SN, use directly
        else:
            predictions[i] = 0
            
    return predictions, probabilities


def evaluate_with_different_thresholds(model, val_images, val_meta, val_labels):
    """
    Evaluate model with different threshold combinations to find best precision/recall tradeoff.
    """
    print("=" * 80)
    print("Evaluating different threshold combinations...")
    print("=" * 80)
    
    threshold_combinations = [
        {'SN': 0.3, 'SLSN-I': 0.5, 'TDE': 0.5},
        {'SN': 0.3, 'SLSN-I': 0.6, 'TDE': 0.5},
        {'SN': 0.3, 'SLSN-I': 0.7, 'TDE': 0.6},
        {'SN': 0.3, 'SLSN-I': 0.8, 'TDE': 0.7},
    ]
    
    for thresholds in threshold_combinations:
        preds, probs = predict_with_confidence_threshold(
            model, val_images, val_meta, thresholds
        )
        
        print(f"\nThresholds: SLSN-I={thresholds['SLSN-I']:.1f}, TDE={thresholds['TDE']:.1f}")
        print(classification_report(val_labels.flatten(), preds, 
                                   target_names=['SN', 'SLSN-I', 'TDE'],
                                   digits=3))
        print("-" * 80)


if __name__ == "__main__":
    print("Precision-Optimized Prediction Module")
    print("Import this module to use confidence-based prediction strategies")

