import tensorflow as tf
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from needle_train.custom_layers import ResNet, DataAugmentation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, MaxPooling2D, Concatenate, Input

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


class FeatureWeightedLayer(Layer):
    def __init__(self, feature_weights, **kwargs):
        super(FeatureWeightedLayer, self).__init__(**kwargs)
        self.feature_weights = tf.constant(feature_weights, dtype=tf.float32)

    def build(self, input_shape):
        # 创建一个可训练的缩放参数
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        super(FeatureWeightedLayer, self).build(input_shape)

    def call(self, inputs):
    
        weighted_inputs = inputs * self.feature_weights
        return weighted_inputs * self.kernel

    def get_config(self):
        config = super(FeatureWeightedLayer, self).get_config()
        config.update({'feature_weights': self.feature_weights.numpy()})
        return config

class TransientClassifier(tf.keras.Model):
    """
    A custom keras model for building a transient CNN-based classifier. 
    ResNet block or plain CNN are both available.
    Metadata are also added.
    """

    def __init__(self, label_dict, N_image, image_dimension, meta_dimension, ks=16, neurons=None, 
                Resnet_op=False, meta_only=False, feature_importance=None, **kwargs):
        super(TransientClassifier, self).__init__(**kwargs)
        
        if neurons is None:
            neurons = [[64, 5], [128, 3]]
        
        self.N_image = N_image
        self.image_dimension = image_dimension
        self.meta_dimension = meta_dimension
        self.ks = ks
        self.neurons = neurons
        self.label_dict = label_dict
        self.Resnet_op = Resnet_op
        self.meta_only = meta_only
        self.feature_importance = feature_importance

        # Data Augmentation Layer
        self.data_augmentation = DataAugmentation()

        # Image processing layers
        if Resnet_op:
            self.res_block = ResNet()
        else:
            self.conv_layers = []
            self.pool_layers = []
            for i, (filters, pool_size) in enumerate(neurons):
                self.conv_layers.append(
                    Conv2D(filters, 3, activation='relu', name=f'conv_{i}')
                )
                self.pool_layers.append(
                    MaxPooling2D((pool_size, pool_size), name=f'pool_{i}')
                )
        
        self.flatten = Flatten()
        # Metadata processing layers
        if self.feature_importance is not None:
            self.meta_weighted = FeatureWeightedLayer(self.feature_importance, name='feature_ranking')
        
        self.dense_m1 = Dense(128, activation='relu', name='dense_me1')
        self.dense_m2 = Dense(128, activation='relu', name='dense_me2')

        # Combined processing layers
        self.concatenate = Concatenate(axis=-1, name='concatenate')
        self.dense_c1 = Dense(256, activation='relu', name='dense_c1')
        self.dense_c2 = Dense(32, activation='relu', name='dense_c2')
        self.output_layer = Dense(len(label_dict), activation='softmax', name='output')
        # self.dropout1 = tf.keras.layers.Dropout(0.3)
        # self.dropout2 = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        image_input = inputs['image_input']
        meta_input = inputs['meta_input']
        
        if not self.meta_only:
            # Process image input
            x = self.data_augmentation(image_input)
          
            
            if self.Resnet_op:
                x = self.res_block(x)
            else:
                for conv, pool in zip(self.conv_layers, self.pool_layers):
                    x = conv(x)
                    x = pool(x)
            
            x = self.flatten(x)
            
            # Process meta input
            if self.feature_importance is not None:
                y = self.meta_weighted(meta_input)
            else:
                y = meta_input
                
            y = self.dense_m1(y)
            y = self.dense_m2(y)
            
            # Combine features
            z = self.concatenate([x, y])
        else:
            # Process only meta input
            if self.feature_importance is not None:
                y = self.meta_weighted(meta_input)
            else:
                y = meta_input
                
            y = self.dense_m1(y)
            y = self.dense_m2(y)
            z = y
        
        # Final processing
        z = self.dense_c1(z)
        z = self.dense_c2(z)
        return self.output_layer(z)

    def get_config(self):
        config = super(TransientClassifier, self).get_config()
        config.update({
            'label_dict': self.label_dict,
            'N_image': self.N_image,
            'image_dimension': self.image_dimension,
            'meta_dimension': self.meta_dimension,
            'ks': self.ks,
            'neurons': self.neurons,
            'Resnet_op': self.Resnet_op,
            'meta_only': self.meta_only,
            'feature_importance': self.feature_importance
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def plot_CM(self, test_images, test_meta, test_labels, save_path, suffix=''):
        predictions = self.predict({'image_input': test_images, 'meta_input': test_meta}, batch_size=32)
        y_pred = np.argmax(predictions, axis=-1)
        y_true = test_labels.flatten()

        labels = list(self.label_dict.keys())
        class_names = [str(label) for label in labels]

        cm = confusion_matrix(y_true, y_pred, labels=list(self.label_dict.values()))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.round(cm_norm, 3)

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(cm_norm, annot=True, ax=ax, fmt='.3f', annot_kws={"size": 20})
          
        ax.set_xlabel('Predicted', fontsize=30)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize=20)
        ax.xaxis.tick_bottom()
        ax.set_ylabel('True', fontsize=30)
        ax.yaxis.set_ticklabels(class_names, fontsize=20)
        ax.tick_params(labelsize=20)
        plt.yticks(rotation=0)

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(os.path.join(save_path, f'cm_{suffix}_{current_time}.png'))
        plt.close()

        return cm

class ROC_AUC_record(tf.keras.callbacks.Callback):
    """
    Custom callback to calculate and record ROC AUC metrics during training.
    This callback calculates ROC AUC for training and validation data at the end of each epoch.
    """
    
    def __init__(self, train_data=None, val_data=None, test_data=None, **kwargs):
        super(ROC_AUC_record, self).__init__(**kwargs)
        self.train_data = train_data  # (X_train, y_train)
        self.val_data = val_data      # (X_val, y_val)
        self.test_data = test_data    # (X_test, y_test)
        
    def on_train_begin(self, logs=None):
        self.ROC_AUC_train_weighted = []
        self.ROC_AUC_train_macro = []
        self.ROC_AUC_val_weighted = []
        self.ROC_AUC_val_macro = []
        self.ROC_AUC_test_weighted = []
        self.ROC_AUC_test_macro = []
        
        # Import sklearn metrics here to avoid import issues
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate ROC AUC for training and validation data at the end of each epoch.
        """
        if self.train_data is not None and self.val_data is not None:
            try:
                # Get predictions for training data
                train_X, train_y = self.train_data
                train_predictions = self.model.predict(train_X, verbose=0)
                
                # Get predictions for validation data
                val_X, val_y = self.val_data
                val_predictions = self.model.predict(val_X, verbose=0)
                
                # Calculate ROC AUC for training data
                try:
                    if len(np.unique(train_y)) > 2:  # Multi-class case
                        train_roc_auc_weighted = self.roc_auc_score(train_y, train_predictions, multi_class='ovr', average='weighted')
                        train_roc_auc_macro = self.roc_auc_score(train_y, train_predictions, multi_class='ovr', average='macro')
                    else:  # Binary case
                        train_roc_auc_weighted = train_roc_auc_macro = self.roc_auc_score(train_y, train_predictions[:, 1])
                except Exception as e:
                    print(f"Error calculating train ROC AUC: {e}")
                    train_roc_auc_weighted = None
                    train_roc_auc_macro = None

                # Calculate ROC AUC for validation data
                try:
                    if len(np.unique(val_y)) > 2:  # Multi-class case
                        val_roc_auc_weighted = self.roc_auc_score(val_y, val_predictions, multi_class='ovr', average='weighted')
                        val_roc_auc_macro = self.roc_auc_score(val_y, val_predictions, multi_class='ovr', average='macro')
                    else:  # Binary case
                        val_roc_auc_weighted = val_roc_auc_macro  = self.roc_auc_score(val_y, val_predictions[:, 1])
                except Exception as e:
                    print(f"Error calculating val ROC AUC: {e}")
                    val_roc_auc_weighted = val_roc_auc_macro = None
                
                # Store the values
                self.ROC_AUC_train_weighted.append(float(train_roc_auc_weighted) if train_roc_auc_weighted is not None else None)
                self.ROC_AUC_train_macro.append(float(train_roc_auc_macro) if train_roc_auc_macro is not None else None)
                self.ROC_AUC_val_weighted.append(float(val_roc_auc_weighted) if val_roc_auc_weighted is not None else None)
                self.ROC_AUC_val_macro.append(float(val_roc_auc_macro) if val_roc_auc_macro is not None else None)
                
                # Add to logs for TensorBoard and other logging systems
                if train_roc_auc_weighted is not None and not np.isnan(train_roc_auc_weighted):
                    logs['roc_auc_train_weighted'] = float(train_roc_auc_weighted)
                if train_roc_auc_macro is not None and not np.isnan(train_roc_auc_macro):
                    logs['roc_auc_train_macro'] = float(train_roc_auc_macro)
                if val_roc_auc_weighted is not None and not np.isnan(val_roc_auc_weighted):
                    logs['roc_auc_val_weighted'] = float(val_roc_auc_weighted)
                if val_roc_auc_macro is not None and not np.isnan(val_roc_auc_macro):
                    logs['roc_auc_val_macro'] = float(val_roc_auc_macro)
                
                print(f"Epoch {epoch + 1}: Train ROC AUC weighted = {train_roc_auc_weighted:.4f}, Train ROC AUC macro = {train_roc_auc_macro:.4f}, Val ROC AUC weighted = {val_roc_auc_weighted:.4f}, Val ROC AUC macro = {val_roc_auc_macro:.4f}")
                
                if self.test_data is not None:
                    try:
                        test_X, test_y = self.test_data
                        test_predictions = self.model.predict(test_X, verbose=0)
                        if len(np.unique(test_y)) > 2:  # Multi-class case
                            test_roc_auc_weighted = self.roc_auc_score(test_y, test_predictions, multi_class='ovr', average='weighted')
                            test_roc_auc_macro = self.roc_auc_score(test_y, test_predictions, multi_class='ovr', average='macro')
                        else:  # Binary case
                            test_roc_auc_weighted = test_roc_auc_macro = self.roc_auc_score(test_y, test_predictions[:, 1])
                        self.ROC_AUC_test_weighted.append(float(test_roc_auc_weighted) if test_roc_auc_weighted is not None else None)
                        self.ROC_AUC_test_macro.append(float(test_roc_auc_macro) if test_roc_auc_macro is not None else None)
                        if test_roc_auc_weighted is not None and not np.isnan(test_roc_auc_weighted):
                            logs['roc_auc_test_weighted'] = float(test_roc_auc_weighted)
                        if test_roc_auc_macro is not None and not np.isnan(test_roc_auc_macro):
                            logs['roc_auc_test_macro'] = float(test_roc_auc_macro)
                        print(f"Epoch {epoch + 1}: Test ROC AUC weighted = {test_roc_auc_weighted:.4f}, Test ROC AUC macro = {test_roc_auc_macro:.4f}")
                  
                    except Exception as e:
                        print(f"Error calculating test ROC AUC: {e}")
                        self.ROC_AUC_test_weighted.append(None)
                        self.ROC_AUC_test_macro.append(None)
                else:
                    self.ROC_AUC_test_weighted.append(None)
                    self.ROC_AUC_test_macro.append(None)
                    # Don't add None values to logs - TensorFlow can't handle them
                
            except Exception as e:
                print(f"Error calculating ROC AUC at epoch {epoch + 1}: {e}")
                # Append None values to maintain list consistency
                self.ROC_AUC_train_weighted.append(None)
                self.ROC_AUC_train_macro.append(None)
                self.ROC_AUC_val_weighted.append(None)
                self.ROC_AUC_val_macro.append(None)
                self.ROC_AUC_test_weighted.append(None)
                self.ROC_AUC_test_macro.append(None)
                # Don't add None values to logs - TensorFlow can't handle them

    # def on_test_end(self, logs=None):
    #     """Called at the end of evaluation."""
        
    #     if self.test_data is not None:
    #         test_X, test_y = self.test_data
    #         test_predictions = self.model.predict(test_X, verbose=0)
    #         if len(np.unique(test_y)) > 2:  # Multi-class case
    #             test_roc_auc = self.roc_auc_score(test_y, test_predictions, multi_class='ovr', average='macro')
    #         else:  # Binary case
    #             test_roc_auc = self.roc_auc_score(test_y, test_predictions[:, 1])
    #         self.ROC_AUC_test.append(float(test_roc_auc))
    #         logs['roc_auc_test'] = float(test_roc_auc)


    def save_to_json(self, file_path):
        """Save ROC AUC values to a JSON file for better readability."""
        import json
        roc_data = {
            "train_roc_auc_weighted": self.ROC_AUC_train_weighted,
            "train_roc_auc_macro": self.ROC_AUC_train_macro,
            "val_roc_auc_weighted": self.ROC_AUC_val_weighted,
            "val_roc_auc_macro": self.ROC_AUC_val_macro,
            "test_roc_auc_weighted": self.ROC_AUC_test_weighted,
            "test_roc_auc_macro": self.ROC_AUC_test_macro,
        }
        with open(file_path, 'w') as f:
            json.dump(roc_data, f, indent=4)

    def plot_ROC_AUC(self, file_path):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(self.ROC_AUC_train_weighted)), self.ROC_AUC_train_weighted, label = 'train_weighted')
        plt.plot(np.arange(len(self.ROC_AUC_train_macro)), self.ROC_AUC_train_macro, label = 'train_macro')
        plt.plot(np.arange(len(self.ROC_AUC_val_weighted)), self.ROC_AUC_val_weighted, label = 'val_weighted')
        plt.plot(np.arange(len(self.ROC_AUC_val_macro)), self.ROC_AUC_val_macro, label = 'val_macro')
        if self.ROC_AUC_test_weighted and len(self.ROC_AUC_test_weighted) > 0:
            plt.plot(np.arange(len(self.ROC_AUC_test_weighted)), self.ROC_AUC_test_weighted, label = 'test_weighted')
            plt.plot(np.arange(len(self.ROC_AUC_test_macro)), self.ROC_AUC_test_macro, label = 'test_macro')
        plt.xlabel('epoch', fontsize = 16)
        plt.ylabel('ROC_AUC', fontsize = 16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend(fontsize = 14)
        plt.savefig(file_path)
        plt.close()

class TrainingHistory(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
 
    def on_train_begin(self, logs=None):
        self.history = {
            'loss': [],
            'val_loss': [],
            'epoch': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Record metrics
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['epoch'].append(epoch)

        # Track best performance
        if logs.get('val_loss') < self.best_val_loss:
            self.best_val_loss = logs.get('val_loss')
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        # Ensure directory exists
        os.makedirs(self.model_name, exist_ok=True)
        
        # Save training history
        np.save(os.path.join(self.model_name, "training_history.npy"), self.history)
        
        # Save best epoch info
        best_performance = {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        np.save(os.path.join(self.model_name, "best_performance.npy"), best_performance)
        
        print(f"\nBest validation loss: {self.best_val_loss:.4f}")
        print(f"Best epoch: {self.best_epoch}")

    def plot_loss(self, file_path):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'], label='train_loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.legend(loc='lower right')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(file_path)
        plt.close()

class F1PerClassMetrics(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='f1_per_class', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = [self.add_weight(name=f"tp_{i}", initializer="zeros") for i in range(num_classes)]
        self.fp = [self.add_weight(name=f"fp_{i}", initializer="zeros") for i in range(num_classes)]
        self.fn = [self.add_weight(name=f"fn_{i}", initializer="zeros") for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        
        for i in range(self.num_classes):
            y_true_i = tf.equal(y_true, i)
            y_pred_i = tf.equal(y_pred, i)
            
            tp = tf.reduce_sum(tf.cast(tf.logical_and(y_true_i, y_pred_i), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true_i), y_pred_i), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(y_true_i, tf.logical_not(y_pred_i)), tf.float32))
            
            self.tp[i].assign_add(tp)
            self.fp[i].assign_add(fp)
            self.fn[i].assign_add(fn)

    def result(self):
        f1s = []
        for i in range(self.num_classes):
            precision = tf.math.divide_no_nan(self.tp[i], self.tp[i] + self.fp[i])
            recall = tf.math.divide_no_nan(self.tp[i], self.tp[i] + self.fn[i])
            f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
            f1s.append(f1)
        
        return tf.reduce_mean(f1s)

    def reset_state(self):
        for var_list in [self.tp, self.fp, self.fn]:
            for var in var_list:
                var.assign(0.0)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_f1s(self):
        return self.f1s

    def save_f1s(self, file_path):
        np.save(file_path, self.f1s)


class RecallPerClassMetrics(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='recall_per_class', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        for i in range(self.num_classes):
            y_true_i = tf.equal(y_true, i)
            y_pred_i = tf.equal(y_pred, i)

            tp = tf.reduce_sum(tf.cast(tf.logical_and(y_true_i, y_pred_i), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(y_true_i, tf.logical_not(y_pred_i)), tf.float32))

            self.tp.assign(tf.tensor_scatter_nd_add(self.tp, [[i]], [tp]))
            self.fn.assign(tf.tensor_scatter_nd_add(self.fn, [[i]], [fn]))

    def result(self):
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        return tf.reduce_mean(recall)  # macro recall

    def reset_state(self):
        for var in [self.tp, self.fn]:
            var.assign(tf.zeros_like(var))

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config

class PrecisionPerClassMetrics(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='precision_per_class', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)

        for i in range(self.num_classes):
            y_true_i = tf.equal(y_true, i)
            y_pred_i = tf.equal(y_pred, i)

            tp = tf.reduce_sum(tf.cast(tf.logical_and(y_true_i, y_pred_i), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true_i), y_pred_i), tf.float32))

      
            self.tp.assign(tf.tensor_scatter_nd_add(self.tp, [[i]], [tp]))
            self.fp.assign(tf.tensor_scatter_nd_add(self.fp, [[i]], [fp]))

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        return tf.reduce_mean(precision)  # macro precision
    
    def reset_state(self):
        for var in [self.tp, self.fp]:
            var.assign(tf.zeros_like(var))

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'num_classes': self.num_classes})
        return base_config


class PerClassMetricsLogger(tf.keras.callbacks.Callback):
    
    def __init__(self, num_classes, save_path=None, val_data=None, test_data=None):
        super().__init__()
        self.num_classes = num_classes
        self.save_path = save_path
        self.val_data = val_data
        self.test_data = test_data
        

        self.history = {
            "precision": {"train": [[] for _ in range(num_classes)], "val": [[] for _ in range(num_classes)], "test": [[] for _ in range(num_classes)]},
            "recall": {"train": [[] for _ in range(num_classes)], "val": [[] for _ in range(num_classes)], "test": [[] for _ in range(num_classes)]},
            "f1": {"train": [[] for _ in range(num_classes)], "val": [[] for _ in range(num_classes)], "test": [[] for _ in range(num_classes)]},
        }


    def _extract_metrics_from_model(self):
        """Extract per-class metrics from the model's metrics without re-predicting."""
        # Find the metrics in the model's compiled metrics
        precision_metric = None
        recall_metric = None
        f1_metric = None
        
        for metric in self.model.metrics:
            if hasattr(metric, 'name'):
     
                if 'precision_per_class' in metric.name:
                    precision_metric = metric
                elif 'recall_per_class' in metric.name:
                    recall_metric = metric
                elif 'f1_per_class' in metric.name:
                    f1_metric = metric
        
        # Extract per-class values from metrics
        precision_per_class = None
        recall_per_class = None
        f1_per_class = None
  
        
        if precision_metric is not None:
            # Extract precision per class from TP and FP
            tp = precision_metric.tp.numpy()
            fp = precision_metric.fp.numpy()
            precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
        

        if recall_metric is not None:
            # Extract recall per class from TP and FN
            tp = recall_metric.tp.numpy()
            fn = recall_metric.fn.numpy()
            recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn)!=0)

        if f1_metric is not None:
            # Extract F1 per class from TP, FP, FN
            tp = np.array([f1_metric.tp[i].numpy() for i in range(self.num_classes)])
            fp = np.array([f1_metric.fp[i].numpy() for i in range(self.num_classes)])
            fn = np.array([f1_metric.fn[i].numpy() for i in range(self.num_classes)])
            
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
            recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
            f1_per_class = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision+recall)!=0)

        return precision_per_class, recall_per_class, f1_per_class

    def _compute_metrics(self, y_true, y_pred):
        """Compute TP, FP, FN per class and return precision, recall, f1 arrays."""
        tp = np.zeros(self.num_classes)
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)

        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        f1 = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            y_true_i = (y_true == i)
            y_pred_i = (y_pred == i)
            tp[i] = np.sum(y_true_i & y_pred_i)
            fp[i] = np.sum(~y_true_i & y_pred_i)
            fn[i] = np.sum(y_true_i & ~y_pred_i)

            # Calculate precision, recall, f1 for this class
            if tp[i] + fp[i] > 0:
                precision[i] = tp[i] / (tp[i] + fp[i])
            else:
                precision[i] = 0.0
                
            if tp[i] + fn[i] > 0:
                recall[i] = tp[i] / (tp[i] + fn[i])
            else:
                recall[i] = 0.0
                
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1[i] = 0.0

        return precision, recall, f1

    def on_epoch_begin(self, epoch, logs=None):
        """Reset metrics at the beginning of each epoch to get epoch-level metrics."""
        # Reset all metrics to get fresh epoch-level calculations
        for metric in self.model.metrics:
            if hasattr(metric, 'reset_state'):
                metric.reset_state()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Extract metrics directly from the model's metrics (no re-prediction needed!)
        precision_per_class, recall_per_class, f1_per_class = self._extract_metrics_from_model()

        for i in range(self.num_classes):
            self.history["precision"]["train"][i].append(precision_per_class[i])
            self.history["recall"]["train"][i].append(recall_per_class[i])
            self.history["f1"]["train"][i].append(f1_per_class[i])
   
        # For validation metrics, we need to compute them separately to avoid mixing with training metrics
        if self.val_data is not None:
            # Extract inputs and labels from the tuple format
            val_inputs, val_labels = self.val_data
            
            # Get validation predictions
            val_predictions = self.model.predict(val_inputs, verbose=0)
            val_pred_classes = np.argmax(val_predictions, axis=-1)
            
            # Compute validation metrics manually
            val_precision, val_recall, val_f1 = self._compute_metrics(val_labels, val_pred_classes)
            
            for i in range(self.num_classes):
                self.history["precision"]["val"][i].append(val_precision[i])
                self.history["recall"]["val"][i].append(val_recall[i])
                self.history["f1"]["val"][i].append(val_f1[i])
                
        if self.test_data is not None:

            test_inputs, test_labels = self.test_data
            test_predictions = self.model.predict(test_inputs, verbose=0)
            test_pred_classes = np.argmax(test_predictions, axis=-1)

            test_precision, test_recall, test_f1 = self._compute_metrics(test_labels, test_pred_classes)
            for i in range(self.num_classes):
                self.history["precision"]["test"][i].append(test_precision[i])
                self.history["recall"]["test"][i].append(test_recall[i])
                self.history["f1"]["test"][i].append(test_f1[i])   

        # Save after each epoch if requested
        if self.save_path:
            np.savez(self.save_path, history=self.history)

    def plot(self, metric, file_path):
        """Plot train/val curves for a given metric (precision/recall/f1)."""
        plt.figure(figsize=(10, 6))
        for i in range(self.num_classes)[1:]:
            plt.plot(self.history[metric]["train"][i], label=f"Train {metric}_class{i}")
            if len(self.history[metric]["val"][i]) > 0:
                plt.plot(self.history[metric]["val"][i], label=f"Val {metric}_class{i}")
            if len(self.history[metric]["test"][i]) > 0:
                plt.plot(self.history[metric]["test"][i], label=f"Test {metric}_class{i}")
        plt.legend(loc='lower right')
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title(f"{metric.upper()} per class")
        plt.savefig(file_path)
        plt.close()

    def save(self, metric, file_path):
        np.save(file_path, self.history[metric])


class SLSNMonitor(tf.keras.callbacks.Callback):
    def __init__(self, val_data, monitor_every=5):
        super().__init__()
        self.val_data = val_data
        self.monitor_every = monitor_every  # Only monitor every N epochs to save memory
        
    def on_epoch_end(self, epoch, logs=None):
        # Skip monitoring if not on monitoring epoch
        if (epoch + 1) % self.monitor_every != 0:
            return
            
        val_X, val_y = self.val_data
        predictions = self.model.predict(val_X, verbose=0, batch_size=32)  # Smaller batch for monitoring
        pred_classes = np.argmax(predictions, axis=-1)
        
        # Focus on SLSN-I (class 1)
        slsn_mask = val_y.flatten() == 1
        if np.sum(slsn_mask) > 0:
            slsn_correct = np.sum((pred_classes[slsn_mask] == 1))
            slsn_total = np.sum(slsn_mask)
            slsn_recall = slsn_correct / slsn_total
            
            # Count what SLSN-I are misclassified as
            slsn_pred = pred_classes[slsn_mask]
            as_sn = np.sum(slsn_pred == 0)
            as_tde = np.sum(slsn_pred == 2)
            
            # Calculate precision: how many predicted SLSN-I are actually SLSN-I?
            predicted_as_slsn = np.sum(pred_classes == 1)
            if predicted_as_slsn > 0:
                slsn_precision = slsn_correct / predicted_as_slsn
            else:
                slsn_precision = 0.0
            
            # F1 score
            if slsn_precision + slsn_recall > 0:
                slsn_f1 = 2 * slsn_precision * slsn_recall / (slsn_precision + slsn_recall)
            else:
                slsn_f1 = 0.0
            
            print(f"\n[SLSN-I Monitor] Epoch {epoch+1}:")
            print(f"  Recall: {slsn_recall:.3f} ({slsn_correct}/{slsn_total})")
            print(f"  Precision: {slsn_precision:.3f} ({slsn_correct}/{predicted_as_slsn})")
            print(f"  F1: {slsn_f1:.3f}")
            print(f"  Misclassified as SN: {as_sn}, as TDE: {as_tde}")
            
            if slsn_recall < 0.05:
                print(f"  ⚠️ WARNING: SLSN-I recall too low!")
            if slsn_precision < 0.20:
                print(f"  ⚠️ WARNING: SLSN-I precision too low! Too many false positives.")

        tde_mask = val_y.flatten() == 2
        if np.sum(tde_mask) > 0:
            tde_correct = np.sum((pred_classes[tde_mask] == 2))
            tde_total = np.sum(tde_mask)
            tde_recall = tde_correct / tde_total
            
            # Count what TDE are misclassified as
            tde_pred = pred_classes[tde_mask]
            as_sn = np.sum(tde_pred == 0)
            as_slsn = np.sum(tde_pred == 1)
            
            # Calculate precision
            predicted_as_tde = np.sum(pred_classes == 2)
            if predicted_as_tde > 0:
                tde_precision = tde_correct / predicted_as_tde
            else:
                tde_precision = 0.0
            
            # F1 score
            if tde_precision + tde_recall > 0:
                tde_f1 = 2 * tde_precision * tde_recall / (tde_precision + tde_recall)
            else:
                tde_f1 = 0.0
            
            print(f"\n[TDE Monitor] Epoch {epoch+1}:")
            print(f"  Recall: {tde_recall:.3f} ({tde_correct}/{tde_total})")
            print(f"  Precision: {tde_precision:.3f} ({tde_correct}/{predicted_as_tde})")
            print(f"  F1: {tde_f1:.3f}")
            print(f"  Misclassified as SN: {as_sn}, as SLSN-I: {as_slsn}")
            
            if tde_recall < 0.05:
                print(f"  ⚠️ WARNING: TDE recall too low!")
            if tde_precision < 0.20:
                print(f"  ⚠️ WARNING: TDE precision too low! Too many false positives.")
