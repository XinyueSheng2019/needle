# from random import shuffle
import pandas as pd 
import numpy as np 
import h5py
import json
import os
# import datetime
# import random

def open_with_h5py(filepath):
    imageset = np.array(h5py.File(filepath, mode = 'r')['imageset'])
    try: 
        labels = np.array(h5py.File(filepath, mode = 'r')['labels'])
    except:
        labels = np.array(h5py.File(filepath, mode = 'r')['label'])
    metaset = np.array(h5py.File(filepath, mode = 'r')['metaset'])
    idx_set = np.array(h5py.File(filepath, mode = 'r')['idx_set'])
    return imageset, labels, metaset, idx_set


def single_untouched_preprocessing(data_path, obj_id, masking = True, has_host = True, scaling_data_path = None, normalize_method = 1):
    if masking:
        label_data_dict = os.path.join(data_path, f'data_dict_untouched_0.npy')
    else:
        label_data_dict = os.path.join(data_path, f'data_dict_untouched_unmasked_0.npy')

    data = np.load(label_data_dict, allow_pickle=True)
    data_dict = dict(data.item()) 


    sci_images = data_dict['imgset'][:,0,:,:]  # Shape (4363, 60, 60)
    sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
    ref_images = data_dict['imgset'][:,1,:,:]  # Shape (4363, 60, 60)
    ref_images = ref_images[..., np.newaxis]
    # Concatenate along a new axis to get shape (4363, 60, 120)
    imageset = np.concatenate([sci_images, ref_images], axis=-1)


    labels = data_dict['labels']
    metaset = data_dict['metaset']
    obj_list = data_dict['obj_match_set'][:,0]

  
    obj_idx = np.where(obj_list == obj_id)[0][0]
    image = imageset[obj_idx]
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[-1])
    label = labels[obj_idx]
    meta = metaset[obj_idx]
    meta = meta.reshape(1, meta.shape[0])

    # Feature reduction
    if has_host:
        meta, _ = feature_reduction_for_mixed_band(meta)
    else:
        meta, _ = feature_reduction_for_mixed_band_no_host(meta)
    
    # Apply scaling - scaling_data_path MUST be provided
    if scaling_data_path is None:
        raise ValueError("scaling_data_path must be provided for untouched data preprocessing. "
                        "Use the scaling_data.json file generated during training data preprocessing.")
    else:
        meta = apply_data_scaling(meta, scaling_data_path, normalize_method)
  
    return image, meta  

    

def single_transient_preprocessing(image, meta):
    image, meta = np.array(image), np.array(meta)
    
    # pre_image = image.reshape(1, image.shape[1], image.shape[-1], image.shape[0])
    pre_meta = meta.reshape(1, meta.shape[0])
    return pre_image, pre_meta


def data_scaling(metaset, output_path, normalize_method = 1):
    if normalize_method == 'normal_by_feature' or normalize_method == 0:
        # normalize by feature
        mt_min = np.nanmin(metaset, axis = 0)
        mt_max = np.nanmax(metaset, axis = 0)
        metaset = (metaset - mt_min)/(mt_max - mt_min)
        s_data = {'max': mt_max.astype('float64').tolist(), 'min': mt_min.astype('float64').tolist()}
    elif normalize_method == 'standarlize_by_feature' or normalize_method == 1:
        # standarlise by feature
        mt_mean = np.nanmean(metaset, axis=0)
        mt_std = np.nanstd(metaset, axis=0)
        metaset = (metaset - mt_mean)/mt_std

        s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}
    elif normalize_method == 'normal_by_sample' or normalize_method == 2:
        # metaset normalization
        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        metaset = (b_metaset-b_min)/(b_max - b_min)
        s_data = {}
    elif normalize_method == 'both' or normalize_method == 3:
        # standarlise by feature
        mt_mean = np.nanmean(metaset, axis=0)
        mt_std = np.nanstd(metaset, axis=0)
        norf_metaset = (metaset - mt_mean)/mt_std
        s_data = {'mean': mt_mean.astype('float64').tolist(), 'std': mt_std.astype('float64').tolist()}

        # metaset normalization
        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        nors_metaset = (b_metaset-b_min)/(b_max - b_min)
        metaset = np.concatenate((norf_metaset, nors_metaset), axis = -1)

    with open(output_path + '/scaling_data.json', 'w') as sd:
            json.dump(s_data, sd, indent = 4)

    return metaset 

def apply_data_scaling(metaset, scaling_file, normalize_method = 1):

    if isinstance(scaling_file, str):
        f = open(scaling_file, 'r')
        scaling = json.load(f)
        f.close()
    else:
        scaling = scaling_file

    if normalize_method == 'normal_by_feature' or normalize_method == 0:
        metaset = (metaset - np.array(scaling['min']))/(np.array(scaling['max']) - np.array(scaling['min']))
    
    elif normalize_method == 'standarlize_by_feature' or normalize_method == 1:
        # standarlise by feature
        # print('DEBUG: mean and std: ', len(np.array(scaling['mean'])), len(np.array(scaling['std'])))
        metaset = (metaset - np.array(scaling['mean']))/np.array(scaling['std'])

    elif normalize_method == 'normal_by_sample' or normalize_method == 2:
        # metaset normalization

        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        metaset = (b_metaset-b_min)/(b_max - b_min)

    elif normalize_method == 'both' or normalize_method == 3:
        # standarlise by feature
        norf_metaset = (metaset - np.array(scaling['mean']))/np.array(scaling['std'])
       
        # metaset normalization
        mt_min = np.nanmin(metaset, axis = 1)[:,np.newaxis]
        mt_max = np.nanmax(metaset, axis = 1)[:,np.newaxis]
        b_metaset, b_min, b_max = np.broadcast_arrays(metaset, mt_min, mt_max)
        nors_metaset = (b_metaset-b_min)/(b_max - b_min)

        metaset = np.concatenate((norf_metaset, nors_metaset), axis = -1)

    return metaset 


def feature_reduction_for_mixed_band(metadata):
    # mixed_nor1_add_disc_t_ext_20240628
    print(metadata.shape)

    feature_names = ['candi_mag_r', 'disc_mag_r', 'delta_mag_discovery_r', 'delta_t_discovery_band_r', 'delta_t_discovery_r', 'ratio_recent_r', 'ratio_disc_r', 'delta_host_mag_r',
                 'candi_mag_g', 'disc_mag_g', 'delta_mag_discovery_g', 'delta_t_discovery_band_g', 'delta_t_discovery_g', 'ratio_recent_g', 'ratio_disc_g', 'delta_host_mag_g',
                  'peak_mag_g_minus_r', 'peak_t_g_minus_r', 
                  'host_g','host_r','host_i','host_z','host_y', 'host_g-r', 'host_r-i', 
                  'offset']
    df = pd.DataFrame(metadata, columns=feature_names)
    df['host_i-z'] = df['host_i'] - df['host_z']
    df['host_z-y'] = df['host_z'] - df['host_y']
    df['ratio_dff_r']  = df['ratio_disc_r'] - df['ratio_recent_r']
    df['ratio_dff_g']  = df['ratio_disc_g'] - df['ratio_recent_g']
    df['disc_mag_g_minus_r'] = df.apply(lambda row: 0 if row['disc_mag_g'] == 0 or row['disc_mag_r'] == 0 else row['disc_mag_g'] - row['disc_mag_r'], axis=1)
    df['colour_dff'] = df.apply(lambda row: 0 if row['peak_mag_g_minus_r'] == 0 or row['disc_mag_g_minus_r'] == 0 else row['peak_mag_g_minus_r'] - row['disc_mag_g_minus_r'], axis=1)
    df['host_tar_colour_g-r'] = df['delta_host_mag_g'] - df['delta_host_mag_r']
    # df = df.drop(['ratio_recent_r', 'ratio_recent_g', 'delta_t_discovery_band_r', 'delta_t_discovery_band_g'], axis = 1)
    # df = df.drop(['peak_t_g_minus_r'], axis = 1) # DEBUG because they are 0 in valid and untouched set, but not in training set.
    return df.to_numpy(), df.columns

def feature_reduction_for_mixed_band_no_host(metadata):
    feature_names = ['candi_mag_r', 'disc_mag_r', 'delta_mag_discovery_r', 'delta_t_discovery_band_r', 'delta_t_discovery_r', 'ratio_recent_r', 'ratio_disc_r',
                 'candi_mag_g', 'disc_mag_g', 'delta_mag_discovery_g', 'delta_t_discovery_band_g', 'delta_t_discovery_g', 'ratio_recent_g', 'ratio_disc_g', 
                  'peak_mag_g_minus_r', 'peak_t_g_minus_r']
    df = pd.DataFrame(metadata, columns=feature_names)
    df['ratio_dff_r']  = df['ratio_disc_r'] - df['ratio_recent_r']
    df['ratio_dff_g']  = df['ratio_disc_g'] - df['ratio_recent_g']
    df['disc_mag_g_minus_r'] = df.apply(lambda row: 0 if row['disc_mag_g'] == 0 or row['disc_mag_r'] == 0 else row['disc_mag_g'] - row['disc_mag_r'], axis=1)
    df['colour_dff'] = df.apply(lambda row: 0 if row['peak_mag_g_minus_r'] == 0 or row['disc_mag_g_minus_r'] == 0 else row['peak_mag_g_minus_r'] - row['disc_mag_g_minus_r'], axis=1)
    # df = df.drop(['ratio_recent_r', 'ratio_recent_g', 'delta_t_discovery_band_r', 'delta_t_discovery_band_g'], axis = 1)
    # df = df.drop(['peak_t_g_minus_r'], axis = 1) # DEBUG because they are 0 in valid and untouched set, but not in training set.
    return df.to_numpy(), df.columns


def save_feature_ranking_plot(xgb_model, feature_names, model_path):
    import matplotlib.pyplot as plt
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': xgb_model.feature_importances_})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.serif': ['Times New Roman'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 8,
        'figure.dpi': 300,
        'figure.autolayout': False
    })
    plt.figure(figsize=(10, 5))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Extreme Gradient Boosting - feature importances (after vetting)')
    plt.xticks(rotation=90)
    plt.savefig(model_path + '/feature_ranking.pdf')


    
def get_feature_ranking(X_train, y_train, class_weight, feature_names, model_path = None, feature_ranking_path = None, save = False):
    if feature_ranking_path is None:
        if model_path is not None:
            import xgboost as xgb
            sample_weights = np.array([class_weight[cls] for cls in y_train])
            best_xgb_params = {'subsample': 0.8, 'n_estimators': 200, 'min_child_weight': 2, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.1, 'colsample_bytree': 0.6}
            xgb_model= xgb.XGBClassifier(**best_xgb_params)
            xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
            if save:
                save_feature_ranking_plot(xgb_model, feature_names, model_path)
            return xgb_model.feature_importances_
        else:
            raise ValueError("model_path is required when feature_ranking_path is not provided")

    else:
        f = open(feature_ranking_path, 'r')
        data = json.load(f)
        f.close()
        return [data[x] for x in feature_names]

    


def get_class_weight(labels):
    class_weight = {}
    for i in np.arange(len(set(labels.flatten()))):
        class_weight[i] = labels.shape[0]/len(np.where(labels.flatten()==i)[0])
    return class_weight


def preprocessing_untouched(filepath, label_dict, output_path, normalize_method = 1, scaling_data_path = None, has_host = True, binary_label = None):
    '''simple preprocessing for running needle2.0, mixed band, with host.
    
    Note: scaling_data_path should always be provided to ensure consistent scaling with training data.
    If None, this function will raise an error since it doesn't create scaling parameters.
    '''
    imageset, labels, metaset, idx_set = open_with_h5py(filepath)
    print(imageset.shape, labels.shape, metaset.shape, idx_set.shape)
    
    # Filter labels BEFORE nan handling (consistent with preprocessing order)
    if has_host:
        label_case = label_dict['label-hosted']
    else:
        label_case = label_dict['label-hostless']

    for k in label_dict['classify'].keys():
        if label_dict['classify'][k] not in label_case.values():
            ab_idx = np.where(labels == label_dict["classify"][k])
            imageset, metaset, labels = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labels, ab_idx, 0)
            idx_set = np.delete(idx_set, ab_idx, 0)

    # if binary_label == 'SLSN-I':
    #     labels[labels==2] = 0
    # elif binary_label == 'TDE':
    #     labels[labels!=2] = 0
    #     labels[labels==2] = 1
     
    # Handle NaN values AFTER label filtering (consistent with preprocessing)
    # imageset = (imageset - np.nanmean(imageset, axis=(1,2), keepdims=True)) / np.nanstd(imageset, axis=(1,2), keepdims=True)


    # TODO: TEST SINGLE OBJECT
    # single_id = 2337
    # single_idx = np.where(idx_set == single_id)[0][0]

    # imageset = imageset[single_idx]
    # metaset = metaset[single_idx]
    # labels = labels[single_idx]
    # imageset = np.expand_dims(imageset, axis=0)
    # metaset = np.expand_dims(metaset, axis=0)
    # labels = np.expand_dims(labels, axis=0)


    imageset = np.nan_to_num(imageset)
    metaset = np.nan_to_num(metaset)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Feature reduction
    if has_host:
        metaset, _ = feature_reduction_for_mixed_band(metaset)
    else:
        metaset, _ = feature_reduction_for_mixed_band_no_host(metaset)
    
    # Apply scaling - scaling_data_path MUST be provided
    if scaling_data_path is None:
        raise ValueError("scaling_data_path must be provided for untouched data preprocessing. "
                        "Use the scaling_data.json file generated during training data preprocessing.")
    else:
        metaset = apply_data_scaling(metaset, scaling_data_path, normalize_method)
  
    # print('TEST SINGLE OBJECT META: ', metaset, np.min(metaset), np.max(metaset), np.mean(metaset), np.std(metaset))
    # print('TEST SINGLE OBJECT IMAGES: ', np.min(imageset), np.max(imageset), np.mean(imageset), np.std(imageset))

    # np.save(output_path + '/test_single_object_imageset.npy', imageset)
    # np.save(output_path + '/test_single_object_metaset.npy', metaset)

    return imageset, metaset, labels, idx_set




def preprocessing(filepath, label_dict, output_path, normalize_method = 1, scaling_data_path = None, feature_ranking_path = None, has_host = True, split_ratio = 0.2, binary_label = None):
    '''simple preprocessing for running needle2.0, mixed band, with host.'''
    imageset, labels, metaset, idx_set = open_with_h5py(filepath)

    if has_host:
        label_case = label_dict['label-hosted']
    else:
        label_case = label_dict['label-hostless']


    for k in label_dict['classify'].keys():
        if label_dict['classify'][k] not in label_case.values():
            ab_idx = np.where(labels == label_dict["classify"][k])
            imageset, metaset, labels = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labels, ab_idx, 0)
            idx_set = np.delete(idx_set, ab_idx, 0)

    # if binary_label == 'SLSN-I':
    #     labels[labels==2] = 0
    # elif binary_label == 'TDE':
    #     labels[labels!=2] = 0
    #     labels[labels==2] = 1
     
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create a random permutation of indices
    permutation = np.random.permutation(len(labels))
    
    # Apply the same random permutation to all arrays
    imageset = imageset[permutation]
    metaset = metaset[permutation] 
    labels = labels[permutation]
    idx_set = idx_set[permutation]

    
    
    if split_ratio > 0 and split_ratio < 1: 
        np.random.seed()
        test_idx = np.random.choice(np.arange(len(labels)), size= int(split_ratio * len(labels)), replace=False) # this is validation set, not test set
        train_imageset, train_metaset, train_labels = np.delete(imageset, test_idx, 0), np.delete(metaset, test_idx, 0), np.delete(labels, test_idx, 0)
        test_imageset, test_metaset, test_labels = np.take(imageset, test_idx, 0), np.take(metaset, test_idx, 0), np.take(labels, test_idx, 0)

        # train_imageset = (train_imageset - np.nanmean(train_imageset, axis=(1,2), keepdims=True)) / np.nanstd(train_imageset, axis=(1,2), keepdims=True)
        # test_imageset = (test_imageset - np.nanmean(test_imageset, axis=(1,2), keepdims=True)) / np.nanstd(test_imageset, axis=(1,2), keepdims=True)


        train_imageset = np.nan_to_num(train_imageset)
        train_metaset = np.nan_to_num(train_metaset)
        test_imageset = np.nan_to_num(test_imageset)
        test_metaset = np.nan_to_num(test_metaset)


        # print('preprocess: ',train_imageset.shape, train_metaset.shape, test_imageset.shape)
        
        if has_host:
            train_metaset, feature_names = feature_reduction_for_mixed_band(train_metaset)
            test_metaset, _ = feature_reduction_for_mixed_band(test_metaset)
        else:
            train_metaset, feature_names = feature_reduction_for_mixed_band_no_host(train_metaset)
            test_metaset, _ = feature_reduction_for_mixed_band_no_host(test_metaset)
        
        XGB_class_weight = get_class_weight(train_labels)
        feature_importances = get_feature_ranking(train_metaset, train_labels, XGB_class_weight, feature_names, output_path, feature_ranking_path)

        if scaling_data_path is None:
            train_metaset = data_scaling(train_metaset, output_path, normalize_method)
            test_metaset = apply_data_scaling(test_metaset, output_path + '/scaling_data.json', normalize_method)
        else:
            train_metaset = apply_data_scaling(train_metaset, scaling_data_path, normalize_method)
            test_metaset = apply_data_scaling(test_metaset, scaling_data_path, normalize_method)
    
        return train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels, feature_importances


    else: # put all data into training set
        test_idx = None
        train_imageset, train_metaset, train_labels = imageset, metaset, labels
        #normalize 
        # train_imageset = (train_imageset - np.nanmean(train_imageset, axis=(1,2), keepdims=True)) / np.nanstd(train_imageset, axis=(1,2), keepdims=True)
        
        train_imageset = np.nan_to_num(train_imageset)
        train_metaset = np.nan_to_num(train_metaset)

        if has_host:
            train_metaset, feature_names = feature_reduction_for_mixed_band(train_metaset)
        else:
            train_metaset, feature_names = feature_reduction_for_mixed_band_no_host(train_metaset)

        XGB_class_weight = get_class_weight(train_labels)
        
        feature_importances = get_feature_ranking(train_metaset, train_labels, XGB_class_weight, feature_names, output_path, feature_ranking_path, save = True)
        
        if scaling_data_path is None:
            train_metaset = data_scaling(train_metaset, output_path, normalize_method)
        else:
            train_metaset = apply_data_scaling(train_metaset, scaling_data_path, normalize_method)

        return train_imageset, train_metaset, train_labels, None, None, None, feature_importances
    
    

    


def select_customised_objs(train_validation_list, reversed_hash):
    '''
    combine a training and a validation set with customized SLSN-I or TDE sets.
    '''
    train_validation_set = {}
    for v in train_validation_list:
        train_validation_set[v] = reversed_hash[v]
    
    return train_validation_set






    

     