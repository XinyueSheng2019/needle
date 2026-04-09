from needle_train.augmentor_pipeline import DataAugmentor
from utils import *
import numpy as np
import os
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Optional, Any
import warnings
from config import *
warnings.filterwarnings("ignore", category=DeprecationWarning, module="extinctions")



def check_data_shape(host_path, batch_id = 0, oversample_list = [0, 100, 200, 500, 1000], hosted = True):
    h = 'hosted_set' if hosted else 'hostless_set'

    for oversample_num in oversample_list:
        data_path = os.path.join(host_path, h, f'oversample_{oversample_num}_{batch_id}.hdf5')
        imageset, metaset, labels, idx_set = open_with_h5py(data_path)
        print(oversample_num)
        print(len(labels[labels==0]), len(labels[labels==1]), len(labels[labels==2]))
       

def convert_data_by_label(input_path, batch_id = 0, oversample_num_dict = {'SLSN-I': 200, 'TDE': 200}, hosted = True):
    '''
    convert the data by label
    '''
    h = 'hosted_set' if hosted else 'hostless_set'
    imageset = []
    labels = []
    metaset = []
    idx_set = []
    idx = 0
    for label in ['SN', 'SLSN-I', 'TDE']:
        if label == 'SN' or oversample_num_dict[label] == 0:
            data_path = os.path.join(input_path, h, label, f'data_dict_train_original_{batch_id}.npy')
            
        else:
            data_path = os.path.join(input_path, h, label, f'data_dict_train_crossmatched_{batch_id}.npy')

        if not os.path.exists(data_path):
            print(f'{label} data unfound for train {batch_id}')
            continue

        data = np.load(data_path, allow_pickle=True)
        data_dict = dict(data.item()) 
        if data_dict['imgset'].shape[0] == 0:
            print(f'no data for train {batch_id} {label}')
            continue
        # Split the image data into science and reference channels
        sci_images = data_dict['imgset'][:,0,:,:]  # Shape (4363, 60, 60)
        sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
        ref_images = data_dict['imgset'][:,1,:,:]  # Shape (4363, 60, 60)
        ref_images = ref_images[..., np.newaxis]
        # Concatenate along a new axis to get shape (4363, 60, 120)
        imgset = np.concatenate([sci_images, ref_images], axis=-1)

        
        if label == 'SN' or oversample_num_dict[label] == 0:
            imageset.append(imgset)
            labels.append(data_dict['labels'])
            metaset.append(data_dict['metaset'])
            idx_set.append(np.arange(idx, idx + data_dict['imgset'].shape[0]))
            idx += data_dict['imgset'].shape[0]
        else:
            imageset.append(imgset[:oversample_num_dict[label]])
            labels.append(data_dict['labels'][:oversample_num_dict[label]])
            metaset.append(data_dict['metaset'][:oversample_num_dict[label]])
            idx_set.append(np.arange(idx, idx + oversample_num_dict[label]))
            idx += oversample_num_dict[label]
    
    for label in ['SLSN-I', 'TDE']:
        original_data_path = os.path.join(input_path, h, label, f'data_dict_train_original_{batch_id}.npy')
        if not os.path.exists(original_data_path):
            print(f'{label} data unfound for original.')
            continue
        original_data = np.load(original_data_path, allow_pickle=True)
        original_data_dict = dict(original_data.item()) 
        if original_data_dict['imgset'].shape[0] == 0:
            print(f'no data for original {batch_id} {h} {label}')
            continue

        sci_images = original_data_dict['imgset'][:,0,:,:]  # Shape (4363, 60, 60)
        sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
        ref_images = original_data_dict['imgset'][:,1,:,:]  # Shape (4363, 60, 60)
        ref_images = ref_images[..., np.newaxis]
        # Concatenate along a new axis to get shape (4363, 60, 120)
        imgset = np.concatenate([sci_images, ref_images], axis=-1)
        imageset.append(imgset)
        labels.append(original_data_dict['labels'])
        metaset.append(original_data_dict['metaset'])
        idx_set.append(np.arange(idx, idx + original_data_dict['imgset'].shape[0]))
        idx += original_data_dict['imgset'].shape[0]
    
    imageset = np.concatenate(imageset, axis=0).astype(np.float32)
    metaset = np.concatenate(metaset, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    idx_set = np.concatenate(idx_set, axis=0).astype(np.int32)

    print(imageset.shape, metaset.shape, labels.shape, idx_set.shape)
    h5_path = os.path.join(input_path, h, f'oversample_custom_slsn_{oversample_num_dict["SLSN-I"]}_tde_{oversample_num_dict["TDE"]}_{batch_id}.hdf5')
    save_to_h5py(imageset, metaset, labels, idx_set, h5_path)
    print(f'oversample_custom_slsn_{oversample_num_dict["SLSN-I"]}_tde_{oversample_num_dict["TDE"]}_{batch_id} {input_path} saved')


def convert_data_by_oversample_num(host_path, batch_id = 0, oversample_num_list = [0, 100, 200, 500, 1000], hosted = True):
    '''
    convert the data by oversample_num
    '''
    h = 'hosted_set' if hosted else 'hostless_set'

    for oversample_num in oversample_num_list:
        imageset = []
        labels = []
        metaset = []
        idx_set = []
        idx = 0
        for c in ['SLSN-I', 'SN', 'TDE']:
            data_path = os.path.join(host_path, h, c, f'data_dict_train_original_{batch_id}.npy')
            if not os.path.exists(data_path):
                print(f'{c} data unfound for original.')
                continue

            data = np.load(data_path, allow_pickle=True)
            data_dict = dict(data.item()) 
        
            if data_dict['imgset'].shape[0] == 0:
                print(f'no data for original {batch_id} {h} {c}')
                continue
        
            # Split the image data into science and reference channels
            sci_images = data_dict['imgset'][:,0,:,:]  # Shape (4363, 60, 60)
            sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
            ref_images = data_dict['imgset'][:,1,:,:]  # Shape (4363, 60, 60)
            ref_images = ref_images[..., np.newaxis]
            # Concatenate along a new axis to get shape (4363, 60, 120)
            imgset = np.concatenate([sci_images, ref_images], axis=-1)
        
            imageset.append(imgset)
            labels.append(data_dict['labels'])
            metaset.append(data_dict['metaset'])
            idx_set.append(np.arange(idx, idx + data_dict['imgset'].shape[0]))
            idx += data_dict['imgset'].shape[0]
            
        if oversample_num > 0:
            for c in ['SLSN-I', 'TDE']:
                data_path = os.path.join(host_path, h, c, f'data_dict_train_crossmatched_{batch_id}.npy')
                if not os.path.exists(data_path):
                    print(f'{c} data unfound for oversample.')
                    continue
                data = np.load(data_path, allow_pickle=True)
                data_dict = dict(data.item()) 
                if data_dict['imgset'].shape[0] == 0:
                    print(f'no data for train_crossmatched {batch_id} {c}')
                    continue
                
                # Split the image data into science and reference channels
                sci_images = data_dict['imgset'][:,0,:,:]  # Shape (4363, 60, 60)
                sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
                ref_images = data_dict['imgset'][:,1,:,:]  # Shape (4363, 60, 60)
                ref_images = ref_images[..., np.newaxis]
                # Concatenate along a new axis to get shape (4363, 60, 120)
                imgset = np.concatenate([sci_images, ref_images], axis=-1)
                if oversample_num > imgset.shape[0]:
                    print(f'oversample_num {oversample_num} is larger than the number of data {imgset.shape[0]} for {batch_id} {h} {c}, use the whole data.')
                    oversample_num = imgset.shape[0]
                imageset.append(imgset[:oversample_num])
                labels.append(data_dict['labels'][:oversample_num])
                metaset.append(data_dict['metaset'][:oversample_num])
                idx_set.append(np.arange(idx, idx + oversample_num))
                idx += oversample_num
                
        imageset = np.concatenate(imageset, axis=0).astype(np.float32)
        metaset = np.concatenate(metaset, axis=0).astype(np.float32)
        labels = np.concatenate(labels, axis=0).astype(np.int32)
        idx_set = np.concatenate(idx_set, axis=0).astype(np.int32)
        h5_path = os.path.join(host_path, h, f'oversample_{oversample_num}_{batch_id}.hdf5')

        print('h: ', h)
        print('oversample_num: ', oversample_num, 'imageset.shape: ', imageset.shape, 'metaset.shape: ', metaset.shape, 'labels.shape: ', labels.shape, 'idx_set.shape: ', idx_set.shape)
        print('SN: ', np.sum(labels == 0), 'SLSN-I: ', np.sum(labels == 1), 'TDE: ', np.sum(labels == 2))
        
        if len(imageset) > 0:
            save_to_h5py(imageset, metaset, labels, idx_set, h5_path)
            print(f'oversample_{oversample_num}_{batch_id} {h} saved')
        else:
            print(f'no data for oversample_{oversample_num}_{batch_id} {h}')


    valid_path = os.path.join(host_path, h, f'valid_{batch_id}.hdf5')
    valid_imageset = []
    valid_metaset = []
    valid_labels = []
    valid_idx_set = []
    idx = 0
    for c in ['SLSN-I', 'SN', 'TDE']:
        data_path = os.path.join(host_path, h, c, f'data_dict_valid_{batch_id}.npy')
        if not os.path.exists(data_path):
            print(f'{c} data unfound for validation.')
            continue
        data = np.load(data_path, allow_pickle=True)
        data_dict = dict(data.item()) 
        if data_dict['imgset'].shape[0] == 0:
            print(f'no data for valid {batch_id} {c}')
            continue
        
        sci_images = data_dict['imgset'][:,0,:,:]  # Shape (4363, 60, 60)
        sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
        ref_images = data_dict['imgset'][:,1,:,:]  # Shape (4363, 60, 60)
        ref_images = ref_images[..., np.newaxis]
        imgset = np.concatenate([sci_images, ref_images], axis=-1)
        valid_imageset.append(imgset)
        valid_metaset.append(data_dict['metaset'])
        valid_labels.append(data_dict['labels'])
        valid_idx_set.append(np.arange(idx, idx + data_dict['imgset'].shape[0]))
        idx += data_dict['imgset'].shape[0]
    valid_imageset = np.concatenate(valid_imageset, axis=0).astype(np.float32)
    valid_metaset = np.concatenate(valid_metaset, axis=0).astype(np.float32)
    valid_labels = np.concatenate(valid_labels, axis=0).astype(np.int32)
    valid_idx_set = np.concatenate(valid_idx_set, axis=0).astype(np.int32)
    save_to_h5py(valid_imageset, valid_metaset, valid_labels, valid_idx_set, valid_path)

    print(f'valid_{batch_id} {h} saved')
    


def get_data_dict(host_path, data_type, batch_id, data_list, h):
    '''
    get the data dict
    '''
    imageset = []
    labels = []
    metaset = []
    idx_set = []
    ztf_id_set = []
    idx = 0
    for c in data_list:
        label_data_path = os.path.join(host_path, h, c, f'data_dict_{data_type}_{batch_id}.npy')
        if not os.path.exists(label_data_path):
            print(f'{label_data_path} not found')
            continue
        data = np.load(label_data_path, allow_pickle=True)
        data_dict = dict(data.item()) 

        if data_dict['imgset'].shape[0] == 0:
            print(f'no data for {data_type} {batch_id} {c}')
            continue
        # Split the image data into science and reference channels
        sci_images = data_dict['imgset'][:,0,:,:]  # Shape (4363, 60, 60)
        sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
        ref_images = data_dict['imgset'][:,1,:,:]  # Shape (4363, 60, 60)
        ref_images = ref_images[..., np.newaxis]
        # Concatenate along a new axis to get shape (4363, 60, 120)
        imgset = np.concatenate([sci_images, ref_images], axis=-1)
        imageset.append(imgset)
        labels.append(data_dict['labels'])
        metaset.append(data_dict['metaset'])
        idx_set.append(np.arange(idx, idx + data_dict['imgset'].shape[0]))
        ztf_id_set.append(data_dict['obj_match_set'][:,0])
        idx += data_dict['imgset'].shape[0]
    imageset = np.concatenate(imageset, axis=0).astype(np.float32)
    metaset = np.concatenate(metaset, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    idx_set = np.concatenate(idx_set, axis=0).astype(np.int32)
    ztf_id_set = np.concatenate(ztf_id_set, axis=0).astype(str)
    # hash table for ztf id to idx
    ztf_id_to_idx = {ztf_id: str(idx) for ztf_id, idx in zip(ztf_id_set, idx_set)}

    with open(os.path.join(host_path, h, f'ztf_id_to_idx_{data_type}_{batch_id}.json'), 'w') as f:
        json.dump(ztf_id_to_idx, f, indent=4)

    return imageset, labels, metaset, idx_set

def convert_unmasked_data(host_path, data_type = 'valid', batch_id = 0, hosted=True):
    '''
    convert the unmasked data
    '''
    h = 'hosted_set' if hosted else 'hostless_set'
    print(f'converting {data_type} {batch_id} {h}')
    if data_type == 'valid':
        c_list = ['SLSN-I', 'SN', 'TDE']
        imageset, labels, metaset, idx_set = get_data_dict(host_path, 'valid_unmasked', batch_id, c_list, h)
        save_path = os.path.join(host_path, h, f'valid_unmasked_{batch_id}.hdf5')
    elif data_type == 'train':
        c_list = ['SLSN-I', 'SN', 'TDE']
        imageset, labels, metaset, idx_set = get_data_dict(host_path, 'train_original_unmasked', batch_id, c_list, h)
        save_path = os.path.join(host_path, h, f'train_original_unmasked_{batch_id}.hdf5')
        
    elif data_type == 'untouched':
        imageset, labels, metaset, idx_set = get_data_dict(host_path, 'untouched_unmasked', batch_id, ['untouched'], h)
        save_path = os.path.join(host_path, h, f'untouched_unmasked_{batch_id}.hdf5')
   
    print('--------------------------------DATASET STATS--------------------------------')
    print('h: ', h)
    print('data_type: ', data_type, 'batch_id: ', batch_id)
    print('SN: ', np.sum(labels == 0), 'SLSN-I: ', np.sum(labels == 1), 'TDE: ', np.sum(labels == 2))
    print('imageset.shape: ', imageset.shape, 'metaset.shape: ', metaset.shape, 'labels.shape: ', labels.shape, 'idx_set.shape: ', idx_set.shape)
    save_to_h5py(imageset, metaset, labels, idx_set, save_path)
    print('--------------------------------DATASET SAVED--------------------------------')


def convert_data(host_path, data_type = 'valid', batch_id = 0, hosted=True):

    
    h = 'hosted_set' if hosted else 'hostless_set'

    print(f'converting {data_type} {batch_id} {h}')


    if data_type == 'untouched':
        imageset, labels, metaset, idx_set = get_data_dict(host_path, data_type, batch_id, ['untouched'], h)
       
    elif data_type == 'valid':
        c_list = ['SLSN-I', 'SN', 'TDE']
        imageset, labels, metaset, idx_set = get_data_dict(host_path, data_type, batch_id, c_list, h)

    elif data_type == 'train':
        c_list = ['SLSN-I', 'SN', 'TDE']
        m_list = ['SLSN-I', 'TDE']

        imageset, labels, metaset, idx_set = get_data_dict(host_path, 'train_original', batch_id, c_list, h)

        save_path = os.path.join(host_path, h, f'train_original_{batch_id}.hdf5')
        save_to_h5py(imageset, metaset, labels, idx_set, save_path)

        imageset2, labels2, metaset2, idx_set2 = get_data_dict(host_path, 'train_crossmatched', batch_id, m_list, h)

        imageset = np.concatenate([imageset, imageset2], axis=0)
        labels = np.concatenate([labels, labels2], axis=0)
        metaset = np.concatenate([metaset, metaset2], axis=0)
        idx_set = np.concatenate([idx_set, idx_set2], axis=0)

    print('--------------------------------DATASET STATS--------------------------------')
    print('h: ', h)
    print('data_type: ', data_type, 'batch_id: ', batch_id)
    print('SN: ', np.sum(labels == 0), 'SLSN-I: ', np.sum(labels == 1), 'TDE: ', np.sum(labels == 2))
    print('imageset.shape: ', imageset.shape, 'metaset.shape: ', metaset.shape, 'labels.shape: ', labels.shape, 'idx_set.shape: ', idx_set.shape)
    h5_path = os.path.join(host_path, h, f'{data_type}_{batch_id}.hdf5')
    save_to_h5py(imageset, metaset, labels, idx_set, h5_path)
    print('--------------------------------DATASET SAVED--------------------------------')


def process_crossmatched_obj(designated_class, img_list, lc_list, masking):
    '''
    process the object
    '''
    DA = DataAugmentor(designated_class=designated_class, display=False, img_list=img_list, lc_list=lc_list, augment = True, masking = masking)
    img_data, upsampled_lc, meta_r, meta_mixed, find_host = DA.next_run_fast(init=True)
    if img_data is None or meta_mixed is None:
        return None, None, None, None, None, None, None, None
    labels = LABEL_DICT[DA.designated_class]
    z = DA.image_preprocessing.img_redshift
    obj_match = [DA.obj_A, DA.obj_B]
    return img_data, upsampled_lc, meta_r, meta_mixed, find_host, labels, z, obj_match


def process_crossmatched_obj_wrapper(args):
    '''
    Wrapper function for multiprocessing
    '''
    designated_class, img_list, lc_list, masking = args
    return process_crossmatched_obj(designated_class, img_list, lc_list, masking)


def process_original_obj(obj, designated_class, masking):

    DA = DataAugmentor(designated_class = designated_class, display = False, augment = False, train_mode = True, masking = masking)
    img_data, upsampled_lc, meta_r, meta_mixed, find_host = DA.naive_run(ztf_id = obj)
                                                                         
    if img_data is None or meta_mixed is None:
        print(f'No data available for ID {obj}')
        return None, None, None, None, None, None, None, None
    
    labels = LABEL_DICT[DA.designated_class]    
  
    obj_match = [obj, obj]
    return img_data, upsampled_lc, meta_r, meta_mixed, find_host, labels, None, obj_match
    
def get_original_train(input_path, designated_class, batch_id, hosted=True):
    '''
    get the original train set
    '''
    if os.path.exists(input_path + f'/{designated_class}_train_{batch_id}_ratio_0.2_hosted_{hosted}.txt'):
        with open(input_path + f'/{designated_class}_train_{batch_id}_ratio_0.2_hosted_{hosted}.txt', 'r') as f:
            train_list = [line.strip() for line in f]
        return train_list
    else:
        print(input_path + f'/{designated_class}_train_{batch_id}_ratio_0.2_hosted_{hosted}.txt', ' unfound, return None.')
        return None


def get_original_valid(input_path, designated_class, batch_id, hosted=True):
    '''
    get the original valid set
    '''
    if os.path.exists(input_path + f'/{designated_class}_valid_{batch_id}_ratio_0.2_hosted_{hosted}.txt'):
        with open(input_path + f'/{designated_class}_valid_{batch_id}_ratio_0.2_hosted_{hosted}.txt', 'r') as f:
            valid_list = [line.strip() for line in f]
        return valid_list
    else:
        print(input_path + f'/{designated_class}_valid_{batch_id}_ratio_0.2_hosted_{hosted}.txt', ' unfound, return None.')
        return None


def get_crossmatched_train(input_path, designated_class, batch_id, hosted=True):
    '''
    get the crossmatched train set from the valid set
    '''
    if os.path.exists(input_path + f'/{designated_class}_train_{batch_id}_ratio_0.2_hosted_{hosted}.txt'):
        with open(input_path + f'/{designated_class}_train_{batch_id}_ratio_0.2_hosted_{hosted}.txt', 'r') as f:
            train_list = [line.strip() for line in f]

        if designated_class == 'SN':
            return train_list, train_list
        else:
            img_list = load_sample_imgs()
            lc_list = load_sample_lc()
            
            good_img_list = []
            good_lc_list = []   
            for i in train_list:
                if i in img_list:
                    good_img_list.append(i)
                if i in lc_list:
                    good_lc_list.append(i)
            return good_img_list, good_lc_list
    else:
        print(input_path + f'/{designated_class}_train_{batch_id}_ratio_0.2_hosted_{hosted}.txt', ' unfound, return None')
        return None, None



def get_valid_train_stratified(split_ratio = 0.2, stratified_sets = 10, output_path = None, hosted = True):
    '''
    get the stratified valid set and train set.
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    obj_dict = {}
    for c in ['TDE', 'SLSN-I', 'SN']:
        objs_list = load_samples(c, hosted = hosted)
        obj_dict[c] = objs_list
   

    for i in range(stratified_sets):
        print(f'stratified_sets {i}')

        for c in ['TDE', 'SLSN-I', 'SN']:
            if int(len(obj_dict[c]) * split_ratio) == 0:
                continue
            valid_list = np.random.choice(obj_dict[c], size=int(len(obj_dict[c]) * split_ratio), replace=False)
            train_list = list(set(obj_dict[c]) - set(valid_list))
            with open(output_path + f'/{c}_valid_{i}_ratio_{split_ratio}_hosted_{hosted}.txt', 'w') as f:
                for obj in valid_list:
                    f.write(obj + '\n') 
            f.close()
            with open(output_path + f'/{c}_train_{i}_ratio_{split_ratio}_hosted_{hosted}.txt', 'w') as f:
                for obj in train_list:
                    f.write(obj + '\n')
            f.close()

def get_valid_train_objs(valid_slsn_num = 10, valid_tde_num = 13, valid_sn_num = 961, output_path = None, batch_id = None, test_mode = False, alert_ratio = True):
    '''
    get the original valid set randomly
    save the valid object ids to a txt file
    These objects will not be used for crossmatched training
    ''' 
    if test_mode:
        valid_slsn_num = 10
        valid_tde_num = 10
        valid_sn_num = 10
        print('------under test mode------') 

    if output_path is None:
        output_path = 'k_fold_sets_new'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for class_type in ['TDE', 'SLSN-I']:
        objs_list = load_samples(class_type)
        print(class_type, len(objs_list))
        valid_list = np.random.choice(objs_list, size=valid_slsn_num if class_type == 'SLSN-I' else valid_tde_num, replace=False)
        print('class_type: ', class_type, 'valid_list: ', len(valid_list))
        with open(output_path + f'/{class_type}_valid_{batch_id}.txt', 'w') as f:
            for obj in valid_list:
                f.write(obj + '\n')
        f.close()
        print(f'{class_type} valid set saved to {output_path}/{class_type}_valid_{batch_id}.txt')

        train_list = list(set(objs_list) - set(valid_list))
        print('class_type: ', class_type, 'train_list: ', len(train_list))
        with open(output_path + f'/{class_type}_train_{batch_id}.txt', 'w') as f:
            for obj in train_list:
                f.write(obj + '\n')
        f.close()

        print(f'{class_type} train set saved to {output_path}/{class_type}_train_{batch_id}.txt')

    if test_mode:
        objs_list = load_samples('SN')[:100] 
    else:
        objs_list = load_samples('SN')

    valid_list = np.random.choice(objs_list, size=valid_sn_num, replace=False)
    with open(output_path + f'/SN_valid_{batch_id}.txt', 'w') as f:
        for obj in valid_list:
            f.write(obj + '\n')
    f.close()
    print(f'SN valid set saved to {output_path}/SN_valid_{batch_id}.txt')

    train_list = list(set(objs_list) - set(valid_list))
    with open(output_path + f'/SN_train_{batch_id}.txt', 'w') as f:
        for obj in train_list:
            f.write(obj + '\n')
    f.close()
    print(f'SN train set saved to {output_path}/SN_train_{batch_id}.txt')



def get_untouched_objs():
    obj_info = pd.read_csv(UNTOUCHED_2025_INFO_PATH)
    obj_list = obj_info['ZTFID'].values
    obj_z_list = obj_info['redshift'].values

    return obj_list, obj_z_list

def process_obj_untouched(ztf_id, z, masking = True) -> tuple:

  
    mag_path = UNTOUCHED_2025_INPUT_IMG_PATH + ztf_id + '/image_meta.json'
    if not os.path.exists(mag_path):
        return None, None, None, None, None, None, None, None
    
    mag_data = json.load(open(mag_path))
    label = RAW_LABEL_DICT['3-class'][mag_data['label']]

    

    print('processing: %s, label: %s' % (ztf_id, label))
    if label != 0 and label != 1 and label!= 2: # remove other classes
        return None, None, None, None, None, None, None, None
    

    print('-------------------', mp.current_process().name, f'processing {ztf_id} -------------------')

    DA = DataAugmentor(obj_A=ztf_id, obj_B=ztf_id, display=False, augment=False, masking = masking, load_img_path=UNTOUCHED_2025_IMG_OUTPUT_PATH if masking else UNTOUCHED_2025_UNMASKED_IMG_OUTPUT_PATH)

    img_data, upsampled_lc, meta_r, meta_mixed, find_host = DA.naive_run(ztf_id=ztf_id, mag_path=UNTOUCHED_2025_MAG_OUTPUT_PATH, 
                                                                         host_data_path=UNTOUCHED_2025_HOST_PATH, 
                                                                         img_path=UNTOUCHED_2025_INPUT_IMG_PATH)
                                                                                 
    
    if img_data is None or meta_mixed is None:
        return None, None, None, None, None, None, None, None


    
    obj_match = [ztf_id, ztf_id]
    return img_data, upsampled_lc, meta_r, meta_mixed, find_host, label, z, obj_match



def process_obj_untouched_wrapper(args):
    '''
    Wrapper function for multiprocessing
    '''
    ztf_id, z, masking = args
    return process_obj_untouched(ztf_id, z, masking)


# Convert lists to NumPy arrays, handling empty cases
def to_numpy_array(data: List, name: str, find_host: bool) -> np.ndarray:
    if not data:
        print(data)
        print(f"No {name} data for {'hosted' if find_host else 'hostless'} set")
        return np.array([])
    try:
        return np.array(data)
    except:
        raise ValueError(f"Failed to convert {name} to NumPy array")



def save_results(
    results_list: List[List[Tuple[Any, Any, Optional[List[float]], Optional[List[float]], bool, Any, Any, Any]]],
    designated_class: str,
    output_name: str,
    input_path: str = 'k_fold_sets_new',
    hosted: bool = True
) -> None:
    """
    Save processed results into hosted and hostless sets for a given class.

    Args:
        results_list: List of result lists containing tuples of (img_data, upsampled_lc, meta_r, meta_mixed, find_host, labels, z, obj_match)
        designated_class: Class name for the results
        output_name: Name for the output file
        input_path: Base directory path for saving results (default: 'k_fold_sets_new')

    Raises:
        ValueError: If meta_mixed lengths are inconsistent or critical data is missing
    """

    h = 'hosted_set' if hosted else 'hostless_set'

    print('saving results for ', designated_class, 'to', input_path)
    # Filter out failed or incomplete results
    results_sum = [
        r for results in results_list 
        for r in results 
        if r[2] is not None and r[3] is not None  # Ensure meta_r and meta_mixed are not None
    ]

    if not results_sum:
        print(f"No valid results for {designated_class}. Skipping save.")
        return

    # Initialize lists for hosted (th) and hostless (t) sets
    imgset, metaset, labelset, z_set, obj_match_set = [], [], [], [], []
    # imgset_t, metaset_t, labels_t, z_set_t, obj_match_set_t = [], [], [], [], []
    count = 0

    for img_data, upsampled_lc, meta_r, meta_mixed, has_host, labels, z, obj_match in results_sum:
        if img_data is None or meta_mixed is None:
            continue
        if has_host != hosted:
            continue

        # Validate meta_mixed length consistency
        # expected_meta_length = 26 if hosted else 16
        # if len(meta_mixed) != expected_meta_length:
            # raise ValueError(f"Inconsistent meta_mixed length for {designated_class}: expected {expected_meta_length}, got {len(meta_mixed)}")
        imgset.append(img_data)
        metaset.append(meta_mixed)
        labelset.append(labels)
        z_set.append(z)
        obj_match_set.append(obj_match)
        count += 1


    print(f'needle-th count for {designated_class}: {count}')


    # Convert to NumPy arrays
    imgset = to_numpy_array(imgset, 'imgset', hosted)
    metaset = to_numpy_array(metaset, 'metaset', hosted)
    labelset = to_numpy_array(labelset, 'labels', hosted)
    z_set = to_numpy_array(z_set, 'z_set', hosted)
    obj_match_set = to_numpy_array(obj_match_set, 'obj_match_set', hosted)
 

    # Create directories
    try:
        os.makedirs(input_path, exist_ok=True)
        needle_set_path = os.path.join(input_path, h, designated_class)
        os.makedirs(needle_set_path, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Failed to create directories: {e}")
  

    # Create data dictionaries
    data_dict = {
        'imgset': imgset,
        'metaset': metaset,
        'labels': labelset,
        'z_set': z_set,
        'obj_match_set': obj_match_set
    }

    # Save data dictionaries
    try:
        np.save(os.path.join(needle_set_path, f'data_dict_{output_name}.npy'), data_dict)
        print(f"Successfully saved results for {designated_class} to {input_path}")
    except Exception as e:
        raise ValueError(f"Failed to save results for {designated_class}: {e}")
    


if __name__ == '__main__':

    input_path = 'up_200_20260228'
    stratified_count = 10
    split_ratio = 0.2
    oversample_num = 200
    masking = True

    # # check_data_shape(host_path = input_path, batch_id = 0, oversample_list = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000], hosted = True)


    # get_valid_train_stratified(split_ratio = split_ratio, stratified_sets = stratified_count, output_path = input_path, hosted = True)
    # get_valid_train_stratified(split_ratio = split_ratio, stratified_sets = stratified_count, output_path = input_path, hosted = False)

    # for i in range(stratified_count):

    #     for hosted in [True, False]:

    #         batch_id = i
    
    #         for label in ['TDE', 'SLSN-I', 'SN']:
    #             valid_list = get_original_valid(input_path = input_path, designated_class = label, batch_id = batch_id, hosted = hosted)
    #             train_list = get_original_train(input_path = input_path, designated_class = label, batch_id = batch_id, hosted = hosted)
                
    #             if valid_list is None or train_list is None:
    #                 continue

    #             worker = partial(process_original_obj, designated_class = label, masking = masking)
    #             with mp.Pool(mp.cpu_count()) as pool:
    #                 results = pool.map(worker, valid_list)

    #             worker2 = partial(process_original_obj, designated_class = label, masking = masking)
    #             with mp.Pool(mp.cpu_count()) as pool:
    #                 results2 = pool.map(worker2, train_list)

    #             if masking:
    #                 save_results([results], designated_class = label, output_name = f'valid_{batch_id}', input_path = input_path, hosted = hosted)
    #                 save_results([results2], designated_class = label, output_name = f'train_original_{batch_id}', input_path = input_path, hosted = hosted)
    #             else:
    #                 save_results([results], designated_class = label, output_name = f'valid_unmasked_{batch_id}', input_path = input_path, hosted = hosted)
    #                 save_results([results2], designated_class = label, output_name = f'train_original_unmasked_{batch_id}', input_path = input_path, hosted = hosted)

    #         if masking:
    #             for label in ['TDE', 'SLSN-I']:
    #                 img_list, lc_list = get_crossmatched_train(input_path = input_path, designated_class=label, batch_id = batch_id, hosted = hosted)
                
    #                 if img_list is None or lc_list is None:
    #                     continue
                    

    #                 # Create arguments for each worker
    #                 worker_args = [(label, img_list, lc_list, masking) for _ in range(oversample_num)]
        
    #                 with mp.Pool(mp.cpu_count()) as pool:
    #                     results = pool.map(process_crossmatched_obj_wrapper, worker_args)
                    
    #                 worker2 = partial(process_original_obj, designated_class=label, masking = masking)
        
    #                 with mp.Pool(mp.cpu_count()) as pool:
    #                     results2 = pool.map(worker2, img_list)


    #                 save_results([results], designated_class=label, output_name = f'train_crossmatched_{batch_id}', input_path = input_path, hosted = hosted)
    #                 save_results([results2], designated_class=label, output_name = f'train_original_{batch_id}', input_path = input_path, hosted = hosted)
        
        
    #             img_list, lc_list = get_crossmatched_train(input_path = input_path, designated_class='SN', batch_id = batch_id, hosted = hosted)
            
    #             worker = partial(process_original_obj, designated_class='SN', masking = masking)
        
    #             with mp.Pool(mp.cpu_count()) as pool:
    #                 results = pool.map(worker, img_list)
                    
    #             save_results([results], designated_class='SN', output_name = f'train_{batch_id}', input_path = input_path)
    #             save_results([results], designated_class='SN', output_name = f'train_original_{batch_id}', input_path = input_path, hosted = hosted)


    #         convert_data(host_path = input_path, data_type = 'valid', batch_id = batch_id, hosted = hosted)
    #         convert_data(host_path = input_path, data_type = 'train', batch_id = batch_id, hosted = hosted)
    #         if not masking: 
    #             convert_unmasked_data(host_path = input_path, data_type = 'valid', batch_id = batch_id, hosted = hosted)
    #             convert_unmasked_data(host_path = input_path, data_type = 'train', batch_id = batch_id, hosted = hosted)

    #         # convert_data_by_oversample_num(host_path = input_path, batch_id = batch_id, oversample_num_list = [0, 100, 200, 300], hosted = hosted)

    for hosted in [True, False]:
        obj_list, obj_z_list = get_untouched_objs()
        batch_id = 0
        worker_args = [(ztf_id, z, masking) for ztf_id, z, masking in zip(obj_list, obj_z_list, [True]*len(obj_list))]
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(process_obj_untouched_wrapper, worker_args)
        if masking:
            save_results([results], designated_class='untouched', output_name = f'untouched_{batch_id}', input_path = input_path, hosted = hosted)
            convert_data(host_path = input_path, data_type = 'untouched', batch_id = batch_id, hosted = hosted)
        else:
            save_results([results], designated_class='untouched', output_name = f'untouched_unmasked_{batch_id}', input_path = input_path, hosted = hosted)
            convert_unmasked_data(host_path = input_path, data_type = 'untouched', batch_id = batch_id, hosted = hosted)
