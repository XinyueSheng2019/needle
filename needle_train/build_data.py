import os
import numpy as np
from config import *
from utils import *


def combine_h5py_data(data_path1, data_path2):
    '''
    combine the data from two paths from hdf5 files.
    '''
    data1 = h5py.File(data_path1, 'r')
    data2 = h5py.File(data_path2, 'r')
    imageset = np.concatenate([data1['imageset'], data2['imageset']], axis=0)
    metaset = np.concatenate([data1['metaset'], data2['metaset']], axis=0)
    labels = np.concatenate([data1['label'], data2['label']], axis=0)
    idx_set = np.concatenate([data1['idx_set'], data2['idx_set']], axis=0)
    return imageset, metaset, labels, idx_set

def combine_np_data(data_path1, data_path2, save = True, save_path = None):
    '''
    combine the data from two paths from .npy files.
    '''
    data1 = np.load(data_path1, allow_pickle=True)
    data2 = np.load(data_path2, allow_pickle=True)
    data1_dict = dict(data1.item())
    data2_dict = dict(data2.item())
    if data1_dict['imgset'].shape[0] == 0 or data2_dict['imgset'].shape[0] == 0:
        print('No data to combine, save empty file')
        return
    imageset = np.concatenate([data1_dict['imgset'], data2_dict['imgset']], axis=0)
    metaset = np.concatenate([data1_dict['metaset'], data2_dict['metaset']], axis=0)
    labels = np.concatenate([data1_dict['labels'], data2_dict['labels']], axis=0)
    z_set = np.concatenate([data1_dict['z_set'], data2_dict['z_set']], axis=0)
    obj_match_set = np.concatenate([data1_dict['obj_match_set'], data2_dict['obj_match_set']], axis=0)
    print('data1: ', data1_dict['imgset'].shape, data1_dict['metaset'].shape, data1_dict['labels'].shape, data1_dict['z_set'].shape, data1_dict['obj_match_set'].shape)
    print('data2: ', data2_dict['imgset'].shape, data2_dict['metaset'].shape, data2_dict['labels'].shape, data2_dict['z_set'].shape, data2_dict['obj_match_set'].shape)
    print('combined: ', imageset.shape, metaset.shape, labels.shape, z_set.shape, obj_match_set.shape)
    if save:
        np.save(save_path, {'imgset': imageset, 'metaset': metaset, 'labels': labels, 'z_set': z_set, 'obj_match_set': obj_match_set})
    return imageset, metaset, labels, z_set, obj_match_set

def load_np_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    data_dict = dict(data.item())
    imageset = data_dict['imgset']
    metaset = data_dict['metaset']
    labels = data_dict['labels']
    z_set = data_dict['z_set']
    obj_match_set = data_dict['obj_match_set']
    return imageset, metaset, labels, z_set, obj_match_set

    
def convert_untouched_data(host_path, data_type = 'original', save = True):
    '''
    convert the untouched data (2024 and 2025 datasets together) to hdf5 file
    '''
    imageset = []
    labels = []
    metaset = []
    idx_set = []
    idx = 0

    label_data_path = os.path.join(host_path, 'untouched', f'data_dict_{data_type}.npy')
    print(label_data_path)
    data = np.load(label_data_path, allow_pickle=True)
    data_dict = dict(data.item()) 
    
    imgset = data_dict['imgset']
    
    imageset.append(imgset)
    labels.append(data_dict['labels'])
    metaset.append(data_dict['metaset'])
    idx_set.append(np.arange(idx, idx + data_dict['imgset'].shape[0]))

    imageset = np.concatenate(imageset, axis=0).astype(np.float32)
    metaset = np.concatenate(metaset, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    idx_set = np.concatenate(idx_set, axis=0).astype(np.int32)

    # Split the image data into science and reference channels
    sci_images = imageset[:,0,:,:]  # Shape (4363, 60, 60)
    sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
    ref_images = imageset[:,1,:,:]  # Shape (4363, 60, 60)
    ref_images = ref_images[..., np.newaxis]
    # Concatenate along a new axis to get shape (4363, 60, 120)
    imageset = np.concatenate([sci_images, ref_images], axis=-1)

    print(imageset.shape, metaset.shape, labels.shape, idx_set.shape)
    print('label_dict: 0: ', labels[labels == 0].shape[0])
    print('label_dict: 1: ', labels[labels == 1].shape[0])
    print('label_dict: 2: ', labels[labels == 2].shape[0])
    if save:
        save_to_h5py(imageset, metaset, labels, idx_set, host_path + f'/untouched_set_{data_type}.hdf5')
    
    return imageset, metaset, labels, idx_set


    
def convert_data(host_path, data_type = 'original', save = True):

    imageset = []
    labels = []
    metaset = []
    idx_set = []
    idx = 0
    for c in ['SLSN-I', 'SN', 'TDE']:
        label_data_path = os.path.join(host_path, c, f'data_dict_{data_type}.npy')
        data = np.load(label_data_path, allow_pickle=True)
        data_dict = dict(data.item()) 
        
        if data_dict['imgset'].shape[0] == 0:
            continue
        # Split the image data into science and reference channels
        sci_images = data_dict['imgset'][:,0,:,:]  # Shape (4363, 60, 60)
        sci_images = sci_images[..., np.newaxis]  # Add channel dimension to make shape (N, 60, 60, 1)
        ref_images = data_dict['imgset'][:,1,:,:]  # Shape (4363, 60, 60)
        ref_images = ref_images[..., np.newaxis]
        # Concatenate along a new axis to get shape (4363, 60, 120)
        imgset = np.concatenate([sci_images, ref_images], axis=-1)
        print(imgset.shape)
        print(data_dict['labels'].shape, data_dict['metaset'].shape,np.arange(idx, idx + data_dict['imgset'].shape[0]).shape )

        imageset.append(imgset)
        labels.append(data_dict['labels'])
        metaset.append(data_dict['metaset'])
        idx_set.append(np.arange(idx, idx + data_dict['imgset'].shape[0]))
   
        idx += data_dict['imgset'].shape[0]

    imageset = np.concatenate(imageset, axis=0).astype(np.float32)
    metaset = np.concatenate(metaset, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    idx_set = np.concatenate(idx_set, axis=0).astype(np.int32)

    if save:
        save_to_h5py(imageset, metaset, labels, idx_set, host_path + f'/training_set_{data_type}.hdf5')
    
    return imageset, metaset, labels, idx_set





def load_data(hosted = True, data_type = 'original'):
    if hosted:
        label_dict = LABEL_DICT_HOSTED
        data_path = os.path.join(NEEDLE_SET_PATH, 'hosted_set', f'training_set_{data_type}.hdf5')
    else:
        label_dict = LABEL_DICT_HOSTLESS
        data_path = os.path.join(NEEDLE_SET_PATH, 'hostless_set', f'training_set_{data_type}.hdf5')

    imageset, metaset, labels, _ = open_with_h5py(data_path)

    if hosted is False:
        labels[labels == 2] = 0 # TDE -> Non-SLSN
    
    return imageset, metaset, labels, label_dict

def load_train_original(hosted = True):
    '''
    build the original training set, without masking or cross-matched
    '''
    imageset, metaset, labels, label_dict = load_data(hosted, 'original')
    print(f"Loaded {imageset.shape[0]} samples")
    return imageset, metaset, labels, label_dict

def load_train_original_mask(hosted = True):
    '''
    build the training set with masking only, including all the samples from the original training set
    '''
    imageset, metaset, labels, label_dict = load_data(hosted, 'original_mask')
    print(f"Loaded {imageset.shape[0]} samples")
    return imageset, metaset, labels, label_dict

def load_train_crossmatched_mask(hosted = True):
    '''
    build the full training set with masking and cross-matched, including all the samples from the original training set and the generated cross-matched set
    '''
    imageset, metaset, labels, label_dict = load_data(hosted, 'original')
    imageset_crossmatched, metaset_crossmatched, labels_crossmatched, label_dict_crossmated = load_data(hosted, 'crossmatched_mask')

    imageset = np.concatenate([imageset, imageset_crossmatched], axis = 0)
    metaset = np.concatenate([metaset, metaset_crossmatched], axis = 0)
    labels = np.concatenate([labels, labels_crossmatched], axis = 0)
    label_dict = {**label_dict, **label_dict_crossmated}
    print(f"Loaded {imageset.shape[0]} samples")
    return imageset, metaset, labels, label_dict


def load_untouched(hosted = True, data_type = 'original'):
    '''
    load the untouched data
    '''
    imageset, metaset, labels, idx_set = convert_untouched_data(hosted, data_type, save = False)
    print(f"Loaded {imageset.shape[0]} samples")
    print('label_dict: 0: ', labels[labels == 0].shape[0])
    print('label_dict: 1: ', labels[labels == 1].shape[0])
    print('label_dict: 2: ', labels[labels == 2].shape[0])
    return imageset, metaset, labels, idx_set

def load_untouched_mask(hosted = True, data_type = 'original_mask'):
    '''
    load the training set with masking only, including all the samples from the original training set
    ''' 
    imageset, metaset, labels, idx_set = convert_untouched_data(hosted, data_type, save = False)
    print(f"Loaded {imageset.shape[0]} samples")
    
    return imageset, metaset, labels, idx_set


def combine_train_valid(data_path, hosted = True):
    '''
    combine the training set and the validation set
    '''
    if hosted:
        train_path = os.path.join(data_path, 'hosted_set', f'train_original_0.hdf5')
        valid_path = os.path.join(data_path, 'hosted_set', f'valid_0.hdf5')
    else:
        train_path = os.path.join(data_path, 'hostless_set', f'train_original_0.hdf5')
        valid_path = os.path.join(data_path, 'hostless_set', f'valid_0.hdf5')
    imageset, metaset, labels, idx_set = combine_h5py_data(train_path, valid_path)

    if hosted:
        save_to_h5py(imageset, metaset, labels, idx_set, data_path + '/hosted_set' + f'/full_set_original.hdf5')
    else:
        save_to_h5py(imageset, metaset, labels, idx_set, data_path + '/hostless_set' + f'/full_set_original.hdf5')

if __name__ == '__main__':

    # combine_train_valid('/Users/xinyuesheng/Documents/astro_projects/scripts/NEEDLE2.0/needle_train/k_fold_sets/', hosted = True)
    
    for t in ['hosted_set', 'hostless_set']:
        host_path = os.path.join(NEEDLE_SET_PATH, t)
        print(host_path)
        dataset, metaset, labels, idx_set = convert_untouched_data(host_path, 'original_mask_2025', save = True)
        save_to_h5py(dataset, metaset, labels, idx_set, host_path + '/test_set_original_mask_2025.hdf5')



