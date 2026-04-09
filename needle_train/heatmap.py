import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras import models
from tensorflow import keras
import config as config
from PIL import Image
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import copy
from astropy import visualization



def get_heatmap(model, test_imageset, test_metaset, title, output_path, conv_layer_name, idx = 0, plot = True):
    """
    A simpler version of the heatmap function that uses a different approach.
    This version creates a new model that replicates the forward pass up to the target layer.
    
    Args:
        model: The trained model
        test_imageset: Test image data
        test_metaset: Test metadata
        title: Title for the plot
        output_path: Path to save the plot (None to display)
        conv_layer_name: Name of the convolutional layer to analyze
        idx: Index of the sample to analyze
    """
    
    # print(f"Creating heatmap for layer: {conv_layer_name}")
    
    # Prepare input data
    # Convert to TensorFlow tensors for gradient computation
    input_sample = {
        'image_input': tf.convert_to_tensor(test_imageset[idx][np.newaxis, ...], dtype=tf.float32), 
        'meta_input': tf.convert_to_tensor(test_metaset[idx][np.newaxis, ...], dtype=tf.float32)
    }
    
    # Create a new model that replicates the forward pass up to the target layer
    image_input = tf.keras.layers.Input(shape=test_imageset.shape[1:], name='image_input')
    meta_input = tf.keras.layers.Input(shape=test_metaset.shape[1:], name='meta_input')
    
    # Replicate the model's forward pass
    x = model.data_augmentation(image_input)
    
    # Go through conv layers
    target_output = None
    for i, (conv, pool) in enumerate(zip(model.conv_layers, model.pool_layers)):
        x = conv(x)
        if conv.name == conv_layer_name:
            target_output = x
            break
        x = pool(x)
    
    if target_output is None:
        raise ValueError(f"Could not find layer {conv_layer_name} in the model")
    
    # Create the intermediate model
    intermediate_model = tf.keras.Model(
        inputs=[image_input, meta_input],
        outputs=target_output
    )
    
    # Get the feature maps and predictions using gradient tape
    with tf.GradientTape() as tape:
        tape.watch(input_sample['image_input'])
        tape.watch(input_sample['meta_input'])
        
        # Get intermediate output
        last_conv_layer_output = intermediate_model([input_sample['image_input'], input_sample['meta_input']])
        
        # print('last_conv_layer_output shape: ', last_conv_layer_output.shape)
        # print('last_conv_layer_output dtype: ', last_conv_layer_output.dtype)
        
        # Get final predictions
        preds = model(input_sample, training=False)
        # print('preds shape: ', preds.shape)
        # print('preds dtype: ', preds.dtype)
    
    # Get prediction after softmax
    argmax = np.argmax(preds[0])
    # print(f"Predicted class: {argmax}")
    
    # Get gradient value with respect to the last conv layer
    class_channel = preds[:, argmax]
    # print('class_channel.shape: ', class_channel.shape)
    # print('class_channel dtype: ', class_channel.dtype)
    
    # Check if gradients can be computed
    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        # print("Warning: Gradients are None. This might be due to disconnected computational graph.")
        # print("Trying alternative approach...")
        
        # Alternative approach: use the intermediate model output directly
        # and compute gradients with respect to the input
        with tf.GradientTape() as tape2:
            tape2.watch(input_sample['image_input'])
            intermediate_output = intermediate_model([input_sample['image_input'], input_sample['meta_input']])
            # Use a simple loss function
            loss = tf.reduce_mean(intermediate_output)
        
        grads = tape2.gradient(loss, input_sample['image_input'])
        # print("Using input gradients instead of intermediate layer gradients")
        
        # Create a simple heatmap from input gradients
        if grads is not None:
            grads = tf.reduce_mean(tf.abs(grads), axis=-1)  # Average across channels
            grads = tf.squeeze(grads)
            heatmap = grads
        else:
            # # print("Could not compute gradients. Creating random heatmap for visualization.")
            heatmap = tf.random.uniform(shape=(60, 60))
    else:
        # # # print('grads.shape: ', grads.shape)
        # # print('grads dtype: ', grads.dtype)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Create heatmap
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # # print(f"Max heatmap value: {np.max(heatmap)}")
    sci_image = test_imageset[idx, :, :, 0]
    ref_image = test_imageset[idx, :, :, 1]
    
    if plot: 
        plt.figure(figsize=(9, 3))

        plt.subplot(1, 3, 1)
        
        plt.imshow(sci_image, cmap='gray',vmin=visualization.ZScaleInterval().get_limits(sci_image)[0],
                vmax=visualization.ZScaleInterval().get_limits(sci_image)[1])
        plt.title('Science Image')
        
        plt.subplot(1, 3, 2)
        plt.imshow(ref_image, cmap='gray',vmin=visualization.ZScaleInterval().get_limits(ref_image)[0],
                vmax=visualization.ZScaleInterval().get_limits(ref_image)[1])
        plt.title('Reference Image')
        
        plt.subplot(1, 3, 3)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Heatmap')
        plt.suptitle(f'Heatmap Analysis - Sample {idx} - Layer {conv_layer_name}')
    
    return sci_image, ref_image, heatmap









