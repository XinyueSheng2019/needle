import os
import random
import warnings
import re

# CRITICAL: Set this BEFORE importing TensorFlow to force CPU-only mode
# This prevents M1 Metal segfaults
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

from config import *
from needle_train.augmentor_pipeline import *
from utils import *
from needle_train.preprocessing import *
from needle_train.training import *
from needle_train.build_data import *
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import models
import numpy as np
import h5py
import tensorflow as tf

import multiprocessing as mp
from multiprocessing import Pool

# Additional Metal GPU disable for M1
try:
    tf.config.set_visible_devices([], 'GPU')
    print("✓ Running in CPU-only mode (M1 Metal disabled)")
except:
    print("✓ GPU config not available (likely CPU-only already)")

from sklearn.metrics import classification_report
warnings.filterwarnings("ignore", category=DeprecationWarning, module="extinctions")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()



custom_objects = {
    'F1PerClassMetrics': F1PerClassMetrics,
    'CustomLearningRateSchedule': CustomLearningRateSchedule,
    'PrecisionPerClassMetrics': PrecisionPerClassMetrics,
    'RecallPerClassMetrics': RecallPerClassMetrics,
    'focal_loss_fixed_modified': focal_loss_modified()
}



def plot_confusion_matrix(y_true, y_pred, output_path, batch_id, original = True, threshold = None, binary_label = None, has_host = True):
    if threshold is not None: 
        cut_results = []
        cut_labels = []
        for result, label in zip(y_pred, y_true):
            if np.max(result) >= threshold:
                cut_results.append(result)
                cut_labels.append(label)

        cut_results = np.array(cut_results)
        cut_labels = np.array(cut_labels)
        
        # Check if any predictions meet the threshold
        if len(cut_results) == 0:
            print(f"Warning: No predictions meet the threshold of {threshold}. Skipping confusion matrix plot.")
            return

        y_pred = np.argmax(cut_results, axis=1)
        y_true = cut_labels
    else:
        y_pred = np.argmax(y_pred, axis=1)

    # Create confusion matrix
    cm_count = confusion_matrix(y_true, y_pred)

    # record the number of each class
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_count, annot=True, annot_kws={"size": 25}, cmap='Blues')
    plt.xlabel('Predicted', fontsize = 25)
    plt.ylabel('True', fontsize = 25)
    plt.title('Confusion Matrix (Count)')

    # Add class labels based on label_dict
    if binary_label is None:
        if has_host:
            class_names = [k for k,v in sorted(RAW_LABEL_DICT['label-hosted'].items(), key=lambda x: x[1])]
        else:
            class_names = [k for k,v in sorted(RAW_LABEL_DICT['label-hostless'].items(), key=lambda x: x[1])] 
    else:
        if binary_label == 'SLSN-I':
            class_names = ['non-SLSN', 'SLSN-I']
        elif binary_label == 'TDE':
            class_names = ['non-TDE', 'TDE']
    plt.xticks(np.arange(len(class_names))+0.5, class_names, rotation=0, fontsize = 20)
    plt.yticks(np.arange(len(class_names))+0.5, class_names, rotation=0, fontsize = 20)

    plt.tight_layout()

    plt.savefig(output_path+'/original_{}_count_{}_threshold_{}_confusion_matrix.png'.format(original, batch_id if batch_id is not None else 'untouched', threshold if threshold is not None else 'None'))
    plt.close()


    cm = cm_count / np.sum(cm_count, axis=1, keepdims=True)

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, annot_kws={"size": 25}, cmap='Blues')
    plt.xlabel('Predicted', fontsize = 25)
    plt.ylabel('True', fontsize = 25)
    plt.title('Confusion Matrix (Completeness)')

    # Add class labels based on label_dict
    if has_host:
        class_names = [k for k,v in sorted(RAW_LABEL_DICT['label-hosted'].items(), key=lambda x: x[1])]
    else:
        class_names = [k for k,v in sorted(RAW_LABEL_DICT['label-hostless'].items(), key=lambda x: x[1])]
    plt.xticks(np.arange(len(class_names))+0.5, class_names, rotation=0, fontsize = 20)
    plt.yticks(np.arange(len(class_names))+0.5, class_names, rotation=0, fontsize = 20)

    plt.tight_layout()

    plt.savefig(output_path+'/original_{}_completeness_{}_threshold_{}_confusion_matrix.png'.format(original, batch_id if batch_id is not None else 'untouched', threshold if threshold is not None else 'None'))
    plt.close()

    # change to purity
    cm = cm_count / (np.sum(cm_count, axis=0, keepdims=True) + 1e-10)

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, annot_kws={"size": 25}, cmap='Blues')
    plt.xlabel('Predicted', fontsize = 25)
    plt.ylabel('True', fontsize = 25)
    plt.title('Confusion Matrix (Purity)')

    # Add class labels based on label_dict
    if binary_label is None:
        if has_host:
            class_names = [k for k,v in sorted(RAW_LABEL_DICT['label-hosted'].items(), key=lambda x: x[1])]
        else:
            class_names = [k for k,v in sorted(RAW_LABEL_DICT['label-hostless'].items(), key=lambda x: x[1])]
    else:
        if binary_label == 'SLSN-I':
            class_names = ['non-SLSN', 'SLSN-I']
        elif binary_label == 'TDE':
            class_names = ['non-TDE', 'TDE']
    plt.xticks(np.arange(len(class_names))+0.5, class_names, rotation=0, fontsize = 20)
    plt.yticks(np.arange(len(class_names))+0.5, class_names, rotation=0, fontsize = 20)

    plt.tight_layout()

    plt.savefig(output_path+'/original_{}_purity_{}_threshold_{}_confusion_matrix.png'.format(original, batch_id if batch_id is not None else 'untouched', threshold if threshold is not None else 'None'))
    plt.close()




def plot_loss(model_path):
    with open(model_path + '/loss_record.txt', 'r') as f:
        loss = f.readlines()
    loss_train = [float(x) for x in loss[0].strip('[]\n').split(',')]
    loss_val = [float(x) for x in loss[1].strip('[]\n').split(',')]
    
    
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(len(loss_train)), loss_train, label = 'train')
    plt.plot(np.arange(len(loss_val)), loss_val, label = 'val')
    plt.xlabel('epoch', fontsize = 16)
    plt.ylabel('loss', fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 14)
    plt.savefig(model_path+'/loss_trend.png')
    plt.close()
 

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid')


def get_loss_record(model_path):
    with open(model_path + '/loss_record.txt', 'r') as f:
        loss = f.readlines()
        loss_train = [float(x) for x in loss[0].strip('[]\n').split(',')]
        loss_val = [float(x) for x in loss[1].strip('[]\n').split(',')]
    return {'loss_train': loss_train, 'loss_val': loss_val}



def find_best_epoch(model_path):
    loss_data = get_loss_record(model_path)
    val_loss = loss_data['loss_val']
    w = 5  # window size
    sm = moving_average(val_loss, w)
    epoch_index = np.argmin(sm) + (w-1)//2 
    best_epoch = epoch_index - (epoch_index % 5) + 5
    return best_epoch


def predict_by_averaged_models(model_list_path, untouched_path, output_path, original, has_host, normalize_method, scaling_data_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_list = []
    for model_path in os.listdir(model_list_path):
        if model_path.startswith('sf_'):
       
            model_path_ = f'{model_list_path}/{model_path}'

            best_epoch = find_best_epoch(model_path_)

            if best_epoch == 5:
                str_best_epoch = '05'
            else:
                str_best_epoch = str(best_epoch)
            if not os.path.exists(os.path.join(model_list_path, model_path, 'smoothed_model_epoch_{}'.format(str_best_epoch))):
                os.makedirs(os.path.join(model_list_path, model_path, 'smoothed_model_epoch_{}'.format(str_best_epoch)))
            TSClassifier = models.load_model(os.path.join(model_list_path, model_path, 'model_epoch_{}'.format(str_best_epoch)), custom_objects=custom_objects)
            _,_ = predict_untouched(TSClassifier, model_path_, untouched_path, model_path_ + '/smoothed_model_epoch_{}'.format(str_best_epoch), original, has_host, None, normalize_method, scaling_data_path, binary_label = None)
 
            model_list.append([TSClassifier, model_path_ + '/smoothed_model_epoch_{}'.format(str_best_epoch)])

    if len(model_list) == 0:
        print(f"Warning: No models found in {model_list_path} starting with 'sf_'")
        return

    averaged_results = []
    for model, model_path in model_list:
        untouched_results, untouched_labels = predict_untouched(model, model_path, untouched_path, output_path, original, has_host, None, normalize_method, scaling_data_path, plot = False, binary_label = None)
        averaged_results.append(untouched_results)

    if len(averaged_results) == 0:
        print("Warning: No results to average")
        return

    averaged_results = np.mean(averaged_results, axis=0)

    # print('averaged_results', averaged_results)

    plot_confusion_matrix(untouched_labels, averaged_results, output_path, batch_id = None, original = original, threshold = None, binary_label = None, has_host = has_host)
    plot_confusion_matrix(untouched_labels, averaged_results, output_path, batch_id = None, original = original, threshold = 0.50, binary_label = None, has_host = has_host)
    plot_confusion_matrix(untouched_labels, averaged_results, output_path, batch_id = None, original = original, threshold = 0.75, binary_label = None, has_host = has_host)
    plot_confusion_matrix(untouched_labels, averaged_results, output_path, batch_id = None, original = original, threshold = 0.90, binary_label = None, has_host = has_host)


         

def predict_untouched(TSClassifier, model_path, untouched_path, output_path, original, has_host, threshold, normalize_method, scaling_data_path, plot = True, binary_label = None):

    untouched_imageset, untouched_metaset, untouched_labels, idx_set = preprocessing_untouched(
        untouched_path,
        RAW_LABEL_DICT,
        model_path,
        normalize_method=normalize_method,
        scaling_data_path=scaling_data_path,
        has_host=has_host,
        binary_label=binary_label,
    )

    untouched_results = TSClassifier.predict({'image_input': untouched_imageset, 'meta_input': untouched_metaset})
    # print('untouched results: ', untouched_results)
    if plot:
        plot_confusion_matrix(untouched_labels, untouched_results, output_path, batch_id = None, original = original, threshold = threshold, binary_label = binary_label, has_host = has_host)
        plot_confusion_matrix(untouched_labels, untouched_results, output_path, batch_id = None, original = original, threshold = 0.5, binary_label = binary_label, has_host = has_host)
        plot_confusion_matrix(untouched_labels, untouched_results, output_path, batch_id = None, original = original, threshold = 0.8, binary_label = binary_label, has_host = has_host)

    y_pred = untouched_results
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print per-object predictions with ZTF ID and true label
    for obj_id, pred_probs, pred_class, true_label in zip(idx_set, y_pred, y_pred_classes, untouched_labels):
        if true_label != 0 and has_host:
            print(f"ZTF ID: {obj_id}, pred_probs: {pred_probs}, pred_class: {pred_class}, true_label: {true_label}")

    # if binary_label is None:
    #     cm_scores = confusion_matrix(untouched_labels, y_pred_classes)
    #     print('untouched confusion matrix: \n', cm_scores)
    #     # print('untouched scores: \n', classification_report(untouched_labels, y_pred_classes, target_names=RAW_LABEL_DICT['label-hosted'].keys()))
    # elif binary_label == 'SLSN-I':
    #     cm_scores = confusion_matrix(untouched_labels, y_pred_classes)
    #     print(f'untouched {binary_label} confusion matrix: \n', cm_scores)
    #     # print('untouched scores: \n', classification_report(untouched_labels, y_pred_classes, target_names=['non-SLSN', 'SLSN-I']))
    # elif binary_label == 'TDE':
    #     cm_scores = confusion_matrix(untouched_labels, y_pred_classes)
    #     print(f'untouched {binary_label} confusion matrix: \n', cm_scores)
    #     # print('untouched scores: \n', classification_report(untouched_labels, y_pred_classes, target_names=['non-TDE', 'TDE']))
    
    return untouched_results, untouched_labels

def run_model(train_path, valid_path, test_path, model_path, output_path, scaling_data_path, normalize_method, threshold, has_host, NEURONS, BATCH_SIZE, EPOCH, LEARNING_RATE, focal_loss_params, batch_id, version, original=False, binary_label = None):


    if not os.path.exists(model_path):
        os.makedirs(model_path)
 
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    train_imageset, train_metaset, train_labels, _,_,_, feature_importances = preprocessing(train_path, RAW_LABEL_DICT, model_path, normalize_method = normalize_method, scaling_data_path = scaling_data_path, feature_ranking_path = None, has_host = has_host, split_ratio = 0, binary_label = binary_label)
    valid_imageset, valid_metaset, valid_labels, _ = preprocessing_untouched(valid_path, RAW_LABEL_DICT, model_path, normalize_method = normalize_method, scaling_data_path = scaling_data_path, has_host = has_host, binary_label = binary_label)
    if test_path is not None:
        test_imageset, test_metaset, test_labels, _ = preprocessing_untouched(test_path, RAW_LABEL_DICT, model_path, normalize_method = normalize_method, scaling_data_path = scaling_data_path, has_host = has_host, binary_label = binary_label)
    else:
        test_imageset = None
        test_metaset = None
        test_labels = None



 
    print('train_imageset.shape, train_metaset.shape, train_labels.shape: ', train_imageset.shape, train_metaset.shape, train_labels.shape)
    print('valid_imageset.shape, valid_metaset.shape, valid_labels.shape: ', valid_imageset.shape, valid_metaset.shape, valid_labels.shape)



    train(train_imageset, train_metaset, train_labels, valid_imageset, valid_metaset, valid_labels, test_imageset, test_metaset, test_labels, feature_importances, RAW_LABEL_DICT, neurons = NEURONS, resnet_op=False, meta_only = False, 
            batch_size = BATCH_SIZE, epoch = EPOCH, learning_rate = LEARNING_RATE, focal_loss_params = focal_loss_params, model_name = model_path, freeze_seed = True)



    TSClassifier = models.load_model(model_path, custom_objects=custom_objects)

    results = TSClassifier.predict({'image_input': valid_imageset, 'meta_input': valid_metaset})

    # print("valid F1 Score:", f1_score(valid_labels, np.argmax(results, axis=1), average='weighted'))
    if binary_label is None:
        # print('valid scores: \n', classification_report(valid_labels, np.argmax(results, axis=1), target_names=RAW_LABEL_DICT['label-hosted'].keys()))
        cm_scores = confusion_matrix(valid_labels, np.argmax(results, axis=1))
        print('valid confusion matrix: \n', cm_scores)
    elif binary_label == 'SLSN-I':
        cm_scores = confusion_matrix(valid_labels, np.argmax(results, axis=1))
        print('valid confusion matrix: \n', cm_scores)
        # print('valid scores: \n', classification_report(valid_labels, np.argmax(results, axis=1), target_names=['non-SLSN', 'SLSN-I']))
    elif binary_label == 'TDE':
        cm_scores = confusion_matrix(valid_labels, np.argmax(results, axis=1))
        print('valid confusion matrix: \n', cm_scores)
        # print('valid scores: \n', classification_report(valid_labels, np.argmax(results, axis=1), target_names=['non-TDE', 'TDE']))
    
    plot_confusion_matrix(valid_labels, results, output_path, batch_id = batch_id, original = original, threshold = threshold, binary_label = binary_label, has_host = has_host)

    plot_loss(model_path)

    best_classifier = models.load_model(model_path + '/best_model/', custom_objects=custom_objects)

    predict_untouched(best_classifier, model_path, test_path, output_path, original, has_host, threshold, normalize_method, scaling_data_path, plot = True, binary_label = binary_label)


def run_model_by_oversample_num(scaling_data_path, data_dir, has_host, NEURONS, BATCH_SIZE, EPOCH, LEARNING_RATE, batch_id, version, oversample_list = None, output_folder = None, model_name = None):
    if oversample_list is None: 
        oversample_list = os.listdir(data_dir)
        oversample_list = [file for file in oversample_list if re.search(r'oversample_\d+', file)]
    # oversample_list = [f'oversample_{num}_{batch_id}.hdf5' for num in [600, 700, 800, 900, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]]
    for oversample_file in oversample_list:
        train_path = os.path.join(data_dir, oversample_file)
        valid_path = os.path.join(data_dir, f'valid_{batch_id}.hdf5')
        if output_folder is None: 
            model_path = f'../models/oversample_compare/{version}/{oversample_file[:-5]}' if model_name is None else f'../models/oversample_compare/{version}/{oversample_file[:-5]}_{model_name}'
        else:
            model_path = f'../models/oversample_compare/{output_folder}/{oversample_file[:-5]}' if model_name is None else f'../models/oversample_compare/{output_folder}/{oversample_file[:-5]}_{model_name}'
        output_path = model_path + '/results'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # else:
        #     print(f'model_path {model_path} already exists')
        #     continue

        save_parameters(model_path, {'version': version, 'd_model': d_model, 'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'epochs': EPOCH, 'original': original, 'has_host': has_host, 'NEURONS': NEURONS, 'batch_id': batch_id, 'threshold': threshold, 'normalize_method': normalize_method, 'scaling_data_path': scaling_data_path})

        run_model(train_path, valid_path, None, model_path, output_path, scaling_data_path, normalize_method, threshold, has_host, NEURONS, BATCH_SIZE, EPOCH, LEARNING_RATE, batch_id = batch_id, version = version)


def train_single_parameter_set(args):
    """
    Train a single parameter set. This function will be called in parallel.
    
    Args:
        args: tuple containing all parameters needed for training
    
    Returns:
        dict: Results including validation loss and parameters
    """
    (train_path, valid_path, test_path, model_path, output_path, scaling_data_path,
     normalize_method, threshold, has_host, neurons, batch_size, lr, version, param_str, original) = args
    
    try:
        print(f"\n[PID {os.getpid()}] Training with parameters: {param_str}")
        print("=" * 80)
        
        # Run model with current parameter combination
        run_model(
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            model_path=model_path,
            output_path=output_path,
            scaling_data_path=scaling_data_path,
            normalize_method=normalize_method,
            threshold=threshold,
            has_host=has_host,
            NEURONS=neurons,
            BATCH_SIZE=batch_size,
            EPOCH=150,
            LEARNING_RATE=lr,
            batch_id=0,
            version=version,
            original=original
        )

        # Load history to get validation loss
        history = np.load(os.path.join(model_path, 'history.npy'), allow_pickle=True).item()
        val_loss = min(history['val_loss'])  # Get best validation loss
        
        print(f"[PID {os.getpid()}] Completed {param_str} - Val Loss: {val_loss:.4f}")
        
        return {
            'param_str': param_str,
            'val_loss': val_loss,
            'model_path': model_path,
            'params': {
                'd_model': neurons[0][0],
                'learning_rate': lr,
                'batch_size': batch_size
            },
            'success': True
        }
        
    except Exception as e:
        print(f"[PID {os.getpid()}] Error in {param_str}: {str(e)}")
        return {
            'param_str': param_str,
            'val_loss': float('inf'),
            'model_path': model_path,
            'params': {
                'd_model': neurons[0][0],
                'learning_rate': lr,
                'batch_size': batch_size
            },
            'success': False,
            'error': str(e)
        }


def grid_search_parameters(train_path, valid_path, test_path, base_model_path, base_output_path, scaling_data_path, version, normalize_method=1, threshold=None, has_host=True, original=False, n_jobs=4):
    """
    Perform grid search over hyperparameters to find optimal model configuration using parallel processing.
    
    Parameters:
    - train_path: Path to training data
    - valid_path: Path to validation data 
    - base_model_path: Base path to save model checkpoints
    - base_output_path: Base path to save results
    - scaling_data_path: Path to scaling data
    - normalize_method: Normalization method to use
    - threshold: Optional threshold parameter
    - has_host: Whether data includes host information
    - original: Whether using original data or crossmatched data
    - n_jobs: Number of parallel jobs to run (default: 4)
    """
    
    import multiprocessing as mp
    from multiprocessing import Pool
    
    # Define parameter grid
    param_grid = {
        'd_model': [64, 128],
        'learning_rates': [8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2],
        'batch_sizes': [32, 64]
    }

    # Generate all parameter combinations
    param_combinations = []
    for d_model in param_grid['d_model']:
        neurons = [[d_model,3], [d_model,3]]
        
        for lr in param_grid['learning_rates']:
            for batch_size in param_grid['batch_sizes']:
                # Create unique model name for this parameter combination
                param_str = f"d{d_model}_lr{lr}_b{batch_size}"
                model_path = os.path.join(base_model_path, f"grid_search_{param_str}")
                
                # Skip if already exists
                if os.path.exists(model_path):
                    print(f"Skipping {param_str} - already exists")
                    continue
                
                # Create directories
                os.makedirs(model_path, exist_ok=True)
                output_path = os.path.join(base_output_path, f"grid_search_{param_str}")
                os.makedirs(output_path, exist_ok=True)
                
                # Prepare arguments for parallel execution
                args = (
                    train_path, valid_path, test_path, model_path, output_path,
                    scaling_data_path, normalize_method, threshold, has_host,
                    neurons, batch_size, lr, version, param_str, original
                )
                param_combinations.append(args)
    
    if not param_combinations:
        print("All parameter combinations already exist. No training needed.")
        return None
    
    print(f"\nStarting parallel grid search with {len(param_combinations)} parameter combinations")
    print(f"Using {min(n_jobs, len(param_combinations))} parallel jobs")
    print("=" * 80)
    
    # Run parallel training
    with Pool(processes=min(n_jobs, len(param_combinations))) as pool:
        results = pool.map(train_single_parameter_set, param_combinations)
    
    # Find best performance
    best_performance = {
        'params': None,
        'model_path': None,
        'val_loss': float('inf'),
        'f1_score': None
    }
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\nGrid Search Complete!")
    print("=" * 80)
    print(f"Successful runs: {len(successful_results)}")
    print(f"Failed runs: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed parameter combinations:")
        for result in failed_results:
            print(f"  {result['param_str']}: {result['error']}")
    
    if successful_results:
        # Find best performing model
        for result in successful_results:
            if result['val_loss'] < best_performance['val_loss']:
                best_performance['val_loss'] = result['val_loss']
                best_performance['params'] = result['params']
                best_performance['model_path'] = result['model_path']
        
        print("\nBest parameters:")
        print(f"d_model: {best_performance['params']['d_model']}")
        print(f"Learning rate: {best_performance['params']['learning_rate']}")
        print(f"Batch size: {best_performance['params']['batch_size']}")
        print(f"Best validation loss: {best_performance['val_loss']}")
        print(f"Best model saved at: {best_performance['model_path']}")
        
        # Print summary of all results
        print(f"\nAll Results Summary:")
        print("-" * 80)
        successful_results.sort(key=lambda x: x['val_loss'])
        for i, result in enumerate(successful_results[:5]):  # Show top 5
            print(f"{i+1}. {result['param_str']}: Val Loss = {result['val_loss']:.4f}")

    return best_performance



def save_parameters(model_path, parameters):
    with open(model_path + '/parameters.txt', 'w') as f:
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")
        f.close()
    print(f"Parameters saved to {model_path}/parameters.txt")

def get_epoch_result(model_name, epoch, version, has_host, normalize_method, scaling_data_path, original, threshold):

    if epoch == 'best': 
        model_path = f'../models/oversample_compare/oversample_optm/oversample_200_0_{model_name}/best_model'
    else:
        model_path = f'../models/oversample_compare/oversample_optm/oversample_200_0_{model_name}/model_epoch_{epoch}'

    TSClassifier = models.load_model(model_path, custom_objects=custom_objects)
    output_path = f'../models/oversample_compare/oversample_optm/oversample_200_0_{model_name}/results_{epoch}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    test_path = f'{version}/hosted_set/untouched_0.hdf5' if has_host else f'{version}/hostless_set/untouched_0.hdf5'
    test_imageset, test_metaset, test_labels, _ = preprocessing_untouched(test_path, RAW_LABEL_DICT, model_path, normalize_method = normalize_method, scaling_data_path = scaling_data_path, has_host = has_host)
    untouched_results = TSClassifier.predict({'image_input': test_imageset, 'meta_input': test_metaset})
    plot_confusion_matrix(test_labels, untouched_results, output_path, batch_id = None, original = original, threshold = threshold)

    valid_path = os.path.join(version, f'hosted_set/valid_0.hdf5')
    valid_imageset, valid_metaset, valid_labels, _ = preprocessing_untouched(valid_path, RAW_LABEL_DICT, model_path, normalize_method = normalize_method, scaling_data_path = scaling_data_path, has_host = has_host)
    valid_results = TSClassifier.predict({'image_input': valid_imageset, 'meta_input': valid_metaset})
    plot_confusion_matrix(valid_labels, valid_results, output_path, batch_id = 0, original = original, threshold = threshold)


def parallel_run(args):
    (output_folder, model_name, train_path, valid_path, test_path, scaling_data_path, normalize_method, threshold, has_host, NEURONS, BATCH_SIZE, EPOCH, LEARNING_RATE, focal_loss_params, batch_id, version, original) = args    

    model_path = f'../models/{output_folder}/{model_name}'
    output_path = model_path + '/results'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    run_model(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        model_path=model_path,
        output_path=output_path,
        scaling_data_path=scaling_data_path,
        normalize_method=normalize_method,
        threshold=threshold,
        has_host=has_host,
        NEURONS=NEURONS,
        BATCH_SIZE=BATCH_SIZE,
        EPOCH=EPOCH,
        LEARNING_RATE=LEARNING_RATE,
        focal_loss_params=focal_loss_params,
        batch_id=batch_id,
        version=version,
        original=original,
        binary_label=None
    )
   
# -----
if __name__ == '__main__':

    
    normalize_method = 1
    threshold = None
    original = False
    has_host = True
    d_model = 128
    masking = True

    NEURONS = [[d_model,3], [d_model,3]] 
    BATCH_SIZE = 64
    EPOCH = 150
    LEARNING_RATE = 8e-5
   
    gamma_list = [0.0, 1.0, 2.5]
    alpha_list = [0.02, 0.49, 0.49]

    # normalize_method = 1
    # threshold = None
    # original = False
    # has_host = False
    # d_model = 128
    # masking = True

    # NEURONS = [[d_model,3], [d_model,3]] 
    # BATCH_SIZE = 32
    # EPOCH = 100
    # LEARNING_RATE = 5e-4
   
    # gamma_list = [0.0, 0.0, 0.0]
    # alpha_list = [0.02, 0.98, 0.0]

    focal_loss_params = {'gamma':gamma_list, 'alpha': alpha_list}
    oversample_num = 200
    version = 'up_200_20260228'

    if masking:
        if original:
            if focal_loss_params is not None:
                output_folder = 'thesis_focal_loss_200_masked_hosted_original_20260315' if has_host else 'thesis_focal_loss_200_masked_hostless_original_20260315'
            else:
                output_folder = 'thesis_no_focal_loss_200_masked_hosted_original_20260315' if has_host else 'thesis_no_focal_loss_200_masked_hostless_original_20260315'
        else:
            if focal_loss_params is not None:
                output_folder = 'thesis_focal_loss_200_masked_hosted_20260315' if has_host else 'thesis_focal_loss_200_masked_hostless_20260315'
            else:
                output_folder = 'thesis_no_focal_loss_200_masked_hosted_20260315' if has_host else 'thesis_no_focal_loss_200_masked_hostless_20260315'
    else:
        if original:
            if focal_loss_params is not None:
                output_folder = 'thesis_focal_loss_200_unmasked_hosted_original_20260315' if has_host else 'thesis_focal_loss_200_unmasked_hostless_original_20260315'
            else:
                output_folder = 'thesis_no_focal_loss_200_unmasked_hosted_original_20260315' if has_host else 'thesis_no_focal_loss_200_unmasked_hostless_original_20260315'
        else:
            if focal_loss_params is not None:
                output_folder = 'thesis_focal_loss_200_unmasked_hosted_20260315' if has_host else 'thesis_focal_loss_200_unmasked_hostless_20260315'
            else:
                output_folder = 'thesis_no_focal_loss_200_unmasked_hosted_20260315' if has_host else 'thesis_no_focal_loss_200_unmasked_hostless_20260315'
    


    worker_args = []
    for i in range(5):
        if has_host: 
            if original: 
                train_path = f'{version}/hosted_set/train_original_{i}.hdf5' if masking else f'{version}/hosted_set/train_original_unmasked_{i}.hdf5'
            else:
                train_path = f'{version}/hosted_set/train_{i}.hdf5' 
            valid_path = f'{version}/hosted_set/valid_{i}.hdf5' if masking else f'{version}/hosted_set/valid_unmasked_{i}.hdf5'
            test_path = f'{version}/hosted_set/untouched_0.hdf5' if masking else f'{version}/hosted_set/untouched_unmasked_0.hdf5'
            scaling_data_path = f'../info/scaling_data_hosted_upsampled.json'
            
        else:
            if original: 
                train_path = f'{version}/hostless_set/train_original_{i}.hdf5' if masking else f'{version}/hostless_set/train_original_unmasked_{i}.hdf5'
            else:
                train_path = f'{version}/hostless_set/train_{i}.hdf5' 

            valid_path = f'{version}/hostless_set/valid_{i}.hdf5' if masking else f'{version}/hostless_set/valid_unmasked_{i}.hdf5'
            test_path = f'{version}/hostless_set/untouched_0.hdf5' if masking else f'{version}/hostless_set/untouched_unmasked_0.hdf5'
            scaling_data_path = f'../info/scaling_data_hostless_upsampled.json'
      

        
        if focal_loss_params is not None:
            model_name = f"sf_{i}_d{d_model}_lr{LEARNING_RATE}_b{BATCH_SIZE}_e{EPOCH}_gamma_{focal_loss_params['gamma'][0]}_{focal_loss_params['gamma'][1]}_{focal_loss_params['gamma'][2]}_alpha_{focal_loss_params['alpha'][0]}_{focal_loss_params['alpha'][1]}_{focal_loss_params['alpha'][2]}"
        else:
            model_name = f"sf_{i}_d{d_model}_lr{LEARNING_RATE}_b{BATCH_SIZE}_e{EPOCH}"
            
        worker_args.append((output_folder, model_name, train_path, valid_path, test_path, scaling_data_path, normalize_method, threshold, has_host, NEURONS, BATCH_SIZE, EPOCH, LEARNING_RATE, focal_loss_params, i, version, original))
   
    with Pool(mp.cpu_count()) as pool:
        pool.map(parallel_run, worker_args)


    # for i in range(10):
    #     # train_path = f'{version}/hosted_set/oversample_{oversample_num}_{i}.hdf5' if masking else f'{version}/hosted_set/oversample_{oversample_num}_unmasked_{i}.hdf5'
    #     train_path = f'{version}/hosted_set/train_original_{i}.hdf5' if masking else f'{version}/hosted_set/train_original_unmasked_{i}.hdf5'
    #     valid_path = f'{version}/hosted_set/valid_{i}.hdf5' if masking else f'{version}/hosted_set/valid_unmasked_{i}.hdf5'
    #     test_path = f'{version}/hosted_set/untouched_0.hdf5' if masking else f'{version}/hosted_set/untouched_unmasked_0.hdf5'
    #     scaling_data_path = f'../info/global_scaling_data_hosted_new.json'
    #     if focal_loss_params is not None:
    #         model_name = f"sf_{i}_d{d_model}_lr{LEARNING_RATE}_b{BATCH_SIZE}_e{EPOCH}_gamma_{focal_loss_params['gamma'][0]}_{focal_loss_params['gamma'][1]}_{focal_loss_params['gamma'][2]}_alpha_{focal_loss_params['alpha'][0]}_{focal_loss_params['alpha'][1]}_{focal_loss_params['alpha'][2]}"
    #     else:
    #         model_name = f"sf_{i}_d{d_model}_lr{LEARNING_RATE}_b{BATCH_SIZE}_e{EPOCH}"

    #     model_path = f'../models/{output_folder}/{model_name}'
    #     output_path = model_path + '/results'

    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)

    #     run_model(
    #         train_path=train_path,
    #         valid_path=valid_path,
    #         test_path=test_path,
    #         model_path=model_path,
    #         output_path=output_path,
    #         scaling_data_path=scaling_data_path,
    #         normalize_method=normalize_method,
    #         threshold=threshold,
    #         has_host=has_host,
    #         NEURONS=NEURONS,
    #         BATCH_SIZE=BATCH_SIZE,
    #         EPOCH=EPOCH,
    #         LEARNING_RATE=LEARNING_RATE,
    #         focal_loss_params=focal_loss_params,
    #         batch_id=i,
    #         version=version,
    #         original=original,
    #         binary_label=binary_label
    #     )

# output_path = '../models/original/new_5_untouched_averaged' if original else '../models/crossmatched/new_5_untouched_averaged'



# model_path = '../models/original/' if original else '../models/crossmatched/'
# model_path = '../models/' + output_folder + '/' + model_name
# predict_by_averaged_models(model_path, output_path, original, has_host, threshold, normalize_method, scaling_data_path)


# # ----- grid search parameters
    # for oversample_num in [200, 1500]:
    #     version = 'up_8000_stratified_1'
    #     train_path = f'{version}/hosted_set/oversample_{oversample_num}_0.hdf5'
    #     valid_path = f'{version}/hosted_set/valid_0.hdf5'
    #     # test_path = 'k_fold_sets_new_5/hosted_set/untouched_0.hdf5' if has_host else 'k_fold_sets_new_5/hostless_set/untouched_0.hdf5'
    #     test_path = None
    #     base_model_path = f'../models/para_search_th/crossmatched_oversample/oversample_{oversample_num}_0/'
    #     base_output_path = base_model_path + '/results'
    #     if not os.path.exists(base_model_path):
    #         os.makedirs(base_model_path)
    #     if not os.path.exists(base_output_path):
    #         os.makedirs(base_output_path)
    #      grid_search_parameters(train_path, valid_path, test_path, base_model_path, base_output_path, scaling_data_path, version, normalize_method, threshold, has_host, original)


#  predict the untouched by designated model
# model_list_path = '../models/telos_focal_loss_0_benchmark_unmasked_hostless_20251118/'
# untouched_path = f'{version}/hostless_set/untouched_unmasked_0.hdf5'

# output_path =  f'{version}/hostless_set/results_averaged_by_telos_focal_loss_0_benchmark_unmasked_hostless_20251118'
# original = False
# has_host = False
# normalize_method = 1
# scaling_data_path = '../info/global_scaling_data_hostless_new.json'
# predict_by_averaged_models(model_list_path, untouched_path, output_path, original, has_host, normalize_method, scaling_data_path)