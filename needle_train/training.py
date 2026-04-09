'''
https://www.tensorflow.org/guide/keras/custom_layers_and_models
'''

# set the seed
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import numpy as np 
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random
from needle_train.transient_model import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="extinctions")

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


def focal_loss_modified(gamma=None, alpha=None):
    '''
    Focal loss with gamma and alpha modification.
    Args:
        gamma: focal loss gamma
        alpha: focal loss alpha
    Returns:
        Loss function callable for model.compile()
    '''
    def focal_loss_fixed_modified(y_true, y_pred):
        # Ensure correct shape and type
        y_true = tf.reshape(y_true, [-1])
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]

        # One-hot encode
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

        # Clip predictions for numerical stability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Compute p_t (prob of true class)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

        # Standard CE part
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)

        # SN modification (only applied to samples with y_true == 0)
        mask_sn = tf.cast(tf.equal(y_true, 0), tf.float32)
        mask_slsn = tf.cast(tf.equal(y_true, 1), tf.float32)
        mask_tde = tf.cast(tf.equal(y_true, 2), tf.float32)


        ce_sn = ce * mask_sn
        ce_slsn = ce * mask_slsn
        ce_tde = ce * mask_tde

        # Alpha weighting (per class)
        if alpha is not None:
            alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)
            alpha_tensor = tf.reshape(alpha_tensor, [-1])
            alpha_sn = tf.gather(alpha_tensor, 0)
            alpha_slsn = tf.gather(alpha_tensor, 1)
            alpha_tde = tf.gather(alpha_tensor, 2)
        else:
            alpha_sn = alpha_slsn = alpha_tde = tf.constant(1.0, dtype=tf.float32)

        # Gamma weighting (per class)
        if gamma is not None:
            gamma_tensor = tf.convert_to_tensor(gamma, dtype=tf.float32)
            gamma_tensor = tf.reshape(gamma_tensor, [-1])
            gamma_sn = tf.gather(gamma_tensor, 0)
            gamma_slsn = tf.gather(gamma_tensor, 1)
            gamma_tde = tf.gather(gamma_tensor, 2)
        else:
            gamma_sn = gamma_slsn = gamma_tde = tf.constant(1.0, dtype=tf.float32)

        # Combine standard CE and per-class focal components
        loss_sn = alpha_sn * tf.pow(1. - p_t, gamma_sn) * ce_sn
        loss_slsn = alpha_slsn * tf.pow(1. - p_t, gamma_slsn) * ce_slsn
        loss_tde = alpha_tde * tf.pow(1. - p_t, gamma_tde) * ce_tde

        loss = loss_sn + loss_slsn + loss_tde

        return tf.reduce_mean(loss)

    return focal_loss_fixed_modified


   

class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, min_lr=1e-6):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.base_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False
        )
    
    def __call__(self, step):
        return tf.math.maximum(self.base_schedule(step), self.min_lr)
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "min_lr": self.min_lr,
        }
   


   
def train(train_imageset, train_metaset, train_labels, valid_imageset, valid_metaset, valid_labels, test_imageset, test_metaset, test_labels, feature_importances, label_dict, neurons= [128,128,128], meta_only = False, batch_size = 32, epoch = 100, learning_rate = 0.00035, focal_loss_params = None, model_name = None, resnet_op=False, freeze_seed = False):
    # Ensure inputs are numpy arrays
    train_imageset = np.array(train_imageset)
    train_metaset = np.array(train_metaset)
    train_labels = np.array(train_labels)
    valid_imageset = np.array(valid_imageset)
    valid_metaset = np.array(valid_metaset)
    valid_labels = np.array(valid_labels)
    if test_imageset is not None:
        test_imageset = np.array(test_imageset)
        test_metaset = np.array(test_metaset)
        test_labels = np.array(test_labels)
    
    print('--------------------------------TRAINING SET STATS--------------------------------')
    print('SN: ', np.sum(train_labels == 0), 'SLSN-I: ', np.sum(train_labels == 1), 'TDE: ', np.sum(train_labels == 2))
    print('--------------------------------VALID SET STATS--------------------------------')
    print('SN: ', np.sum(valid_labels == 0), 'SLSN-I: ', np.sum(valid_labels == 1), 'TDE: ', np.sum(valid_labels == 2))
    print('--------------------------------TEST SET STATS--------------------------------')
    print('SN: ', np.sum(test_labels == 0), 'SLSN-I: ', np.sum(test_labels == 1), 'TDE: ', np.sum(test_labels == 2))
    print('train_imageset.shape: ', train_imageset.shape, 'train_metaset.shape: ', train_metaset.shape, 'train_labels.shape: ', train_labels.shape)
    print('valid_imageset.shape: ', valid_imageset.shape, 'valid_metaset.shape: ', valid_metaset.shape, 'valid_labels.shape: ', valid_labels.shape)
    print('test_imageset.shape: ', test_imageset.shape, 'test_metaset.shape: ', test_metaset.shape, 'test_labels.shape: ', test_labels.shape)



    # Initialize ROC AUC logger with training and validation data
    train_data = ({'image_input': train_imageset, 'meta_input': train_metaset}, train_labels)
    val_data = ({'image_input': valid_imageset, 'meta_input': valid_metaset}, valid_labels)

    if test_imageset is not None:
        test_data = ({'image_input': test_imageset, 'meta_input': test_metaset}, test_labels)
    else:
        test_data = None


    current_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Set model path
    if model_name is None:
        model_path = 'models/models_nor_' + current_time
    else:
        model_path = model_name

    # Ensure directory exists
    os.makedirs(model_path, exist_ok=True)

    # Create a custom learning rate schedule class
     # check label_dict
    if len(set(train_labels.flatten())) == 2:
        print('hostless case, down to 2 classes.')
        label_dict = label_dict['label-hostless']
        num_classes = 2
    else:
        label_dict = label_dict['label-hosted']
        num_classes = 3
    
    if resnet_op is False:
        TCModel = TransientClassifier(label_dict, N_image = 60, image_dimension = train_imageset.shape[-1], meta_dimension = train_metaset.shape[-1], neurons = neurons, meta_only= meta_only, feature_importance=feature_importances)
    else:
        TCModel = TransientClassifier(label_dict, N_image = 60, image_dimension = train_imageset.shape[-1],  meta_dimension = train_metaset.shape[-1], neurons = neurons, Resnet_op = True, feature_importance=feature_importances)
   
      
    lr_schedule = CustomLearningRateSchedule(
        initial_learning_rate=learning_rate,
        decay_steps=train_imageset.shape[0]/batch_size, 
        decay_rate=0.96,  # Less aggressive
        min_lr=2e-6
)


    # Create ModelCheckpoint callback to save best model
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_path, 'best_model'),
        monitor='val_loss',
        save_best_only=True,
        save_format='tf',  
        mode='min',
        verbose=1
    )
    

    checkpoint_path = os.path.join(model_path, 'model_epoch_{epoch:02d}')
    checkpoint_periodic = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_format='tf',  
        save_freq='epoch',
        period=5,
        verbose=1
    )


    # TEMPORARILY DISABLED for M1 Mac stability - these are memory intensive
    ROC_AUC_logger = ROC_AUC_record(train_data=train_data, val_data=val_data, test_data=test_data)
    training_history = TrainingHistory(model_path)
    logger = PerClassMetricsLogger(num_classes=num_classes, val_data=val_data, test_data=test_data)
    
    
    TCModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
        loss = focal_loss_modified(gamma=focal_loss_params['gamma'], alpha=focal_loss_params['alpha']) if focal_loss_params is not None else tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [PrecisionPerClassMetrics(num_classes=num_classes), 
        RecallPerClassMetrics(num_classes=num_classes), 
        F1PerClassMetrics(num_classes=num_classes)]
        )
    


    class_weight = {0: 0.02, 1: 0.49, 2: 0.49}

    if freeze_seed: 
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'image_input': train_imageset, 'meta_input': train_metaset}, train_labels)
        )
        train_dataset = train_dataset.shuffle(buffer_size=len(train_labels), seed=8)  # 固定种子
        train_dataset = train_dataset.batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'image_input': valid_imageset, 'meta_input': valid_metaset}, valid_labels)
        ).batch(batch_size)
    
      
        fit_history = TCModel.fit(
            train_dataset,
            epochs=epoch,
            callbacks=[
                checkpoint_periodic, 
                checkpoint, 
                training_history,  
                ROC_AUC_logger,  # DISABLED for M1
                logger
                ],
            class_weight=class_weight,
            validation_data= ({'image_input': valid_imageset, 'meta_input': valid_metaset}, valid_labels),
            workers=1,
            use_multiprocessing=False
        )

        try:
            TCModel.evaluate({'image_input': valid_imageset, 'meta_input': valid_metaset}, valid_labels)
        except Exception as e:
            print(f"Evaluation failed with error: {e}")
    else:
        fit_history = TCModel.fit(
                {'image_input': train_imageset, 'meta_input': train_metaset}, train_labels,
                shuffle=True,
                epochs=epoch,
                batch_size=batch_size,
                callbacks=[
                    checkpoint_periodic, 
                    checkpoint, 
                    training_history, 
                    ROC_AUC_logger,  # DISABLED for M1
                    logger
                ],
                class_weight=class_weight,
                validation_data=(
                    {'image_input': valid_imageset, 'meta_input': valid_metaset}, valid_labels
                ),
                use_multiprocessing=False  # Changed from True for M1 Mac stability
            )

        TCModel.evaluate({'image_input': valid_imageset, 'meta_input': valid_metaset}, valid_labels)

    try:
        TCModel.save(model_path, save_format='tf')
        print(f"Model saved successfully to {model_path} in TensorFlow format")
    except Exception as e:
        print(f"Model saving failed with error: {e}")
        try:
            TCModel.save_weights(os.path.join(model_path, 'model_weights'))
            print(f"Model weights saved to {model_path}/model_weights")
        except Exception as e2:
            print(f"Weight saving also failed: {e2}")
    try:
        cm = TCModel.plot_CM(valid_imageset, valid_metaset, valid_labels, save_path = model_path)
    except Exception as e:
        print(f"Confusion matrix plotting failed with error: {e}")
        cm = None
    
    with open(os.path.join(model_path, 'loss_record.txt'), 'w') as f:
        f.write(str(fit_history.history['loss'])+'\n'+str(fit_history.history['val_loss'])+'\n')
    
    # Save training history as numpy file for grid search
    np.save(os.path.join(model_path, 'history.npy'), fit_history.history)
    training_history.plot_loss(os.path.join(model_path, 'loss_trend.pdf'))

    # DISABLED for M1 stability - re-enable these if needed after training works
    ROC_AUC_logger.plot_ROC_AUC(os.path.join(model_path, 'ROC_AUC_trend.pdf'))
    logger.plot(metric='recall', file_path=os.path.join(model_path, 'Recall_trend.pdf'))
    logger.plot(metric='precision', file_path=os.path.join(model_path, 'Precision_trend.pdf'))
    logger.plot(metric='f1', file_path=os.path.join(model_path, 'F1_trend.pdf'))
    

    ROC_AUC_logger.save_to_json(os.path.join(model_path, 'ROC_AUC_record.json'))
    logger.save(metric='f1', file_path=os.path.join(model_path, 'F1_record.npy'))
    logger.save(metric='recall', file_path=os.path.join(model_path, 'Recall_record.npy'))
    logger.save(metric='precision', file_path=os.path.join(model_path, 'Precision_record.npy'))





        

