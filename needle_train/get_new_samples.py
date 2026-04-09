from needle_train.augmentor_pipeline import DataAugmentor
from utils import load_samples
from config import NEEDLE_SET_PATH, LABEL_DICT, RAW_LABEL_DICT
import json
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from functools import partial

'''
THIS IS AN OLD FILE, REPLACED WITH 'get_train_valid_sets.py'.
'''


def process_obj_crossmatched(sample_id: int, designated_class) -> tuple:
 
    print('-------------------', mp.current_process().name, f'processing id {sample_id} -------------------')
    DA = DataAugmentor(designated_class=designated_class, display=False)
    img_data, upsampled_lc, meta_r, meta_mixed, find_host = DA.next_run_fast(init=True)
    if img_data is None:
        return None, None, None, None, None, None, None, None
    labels = LABEL_DICT[DA.designated_class]
    z = DA.image_preprocessing.img_redshift

    obj_match = [DA.obj_A, DA.obj_B]
    return img_data, upsampled_lc, meta_r, meta_mixed, find_host, labels, z, obj_match



def process_obj_SN(sample_id: int, obj_list, designated_class) -> tuple:
    # try:
    print('-------------------', mp.current_process().name, f'processing id {sample_id}: {obj_list[sample_id]} -------------------')

    DA = DataAugmentor(designated_class=designated_class, display=False, augment=False)

    ztf_id = obj_list[sample_id]
    img_data, upsampled_lc, meta_r, meta_mixed, find_host = DA.naive_run(ztf_id=ztf_id)

    if img_data is None:
        print(f'No data available for ID {ztf_id}')
        return None, None, None, None, None, None, None, None
    
    labels = LABEL_DICT[DA.designated_class]
    # z = DA.image_preprocessing.img_redshift

    obj_match = [ztf_id]
    return img_data, upsampled_lc, meta_r, meta_mixed, find_host, labels, None, obj_match
    
    # except Exception as e:
    #     print(f"Error processing ID {sample_id}: {e}")
    #     return None, None, None, None, None, None, None, None


def process_obj_untouched(ztf_id) -> tuple:
    untouched_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/NEEDLE2.0/untouched_2025/'
    untouched_hosted_path = untouched_path + 'hosts_ext/'
    untouched_mag_path = untouched_path + 'mags/'
    untouched_img_path = untouched_path + 'images/'
    untouched_obj_info_path = untouched_path + '20240225_20250603.csv'
    processed_img_path = untouched_path + 'image_preprocessing_output/'
    obj_info = pd.read_csv(untouched_obj_info_path)

    mag_path = untouched_img_path + ztf_id + '/image_meta.json'
    mag_data = json.load(open(mag_path))
    label = RAW_LABEL_DICT['3-class'][mag_data['label']]
    print('processing: %s, label: %s' % (ztf_id, label))
    if label == 3: # remove other classes
        return None, None, None, None, None, None, None, None

    try: 
        z = obj_info[obj_info['ZTFID'] == ztf_id]['redshift'].values[0]
    except:
        z = None

    del obj_info

    obj_match = [ztf_id]


    print('-------------------', mp.current_process().name, f'processing {ztf_id} -------------------')

    DA = DataAugmentor(obj_A=ztf_id, obj_B=ztf_id, display=False, augment=False)

    img_data, upsampled_lc, meta_r, meta_mixed, find_host = DA.naive_run(ztf_id=ztf_id, mag_path=untouched_mag_path, 
                                                                         host_data_path=untouched_hosted_path, 
                                                                         img_path=untouched_img_path,
                                                                         load_img_path=processed_img_path
                                                                         )
    
    if img_data is None:
        return None, None, None, None, None, None, None, None

    return img_data, upsampled_lc, meta_r, meta_mixed, find_host, label, z, obj_match


def save_results(results, designated_class, output_name):
 # # Filter out failed or incomplete results
    results = [r for r in results if r[0] is not None and r[3] is not None]

    imgset_th, metaset_th, labels_th, z_set_th, obj_match_set_th = [], [], [], [], []
    imgset_t, metaset_t, labels_t, z_set_t, obj_match_set_t = [], [], [], [], []

    count_th = 0
    count_t = 0

    for img_data, upsampled_lc, meta_r, meta_mixed, find_host, labels, z, obj_match in results:
        if img_data is None:
            continue
        if find_host:
            imgset_th.append(img_data)
            metaset_th.append(meta_mixed)
            labels_th.append(labels)
            z_set_th.append(z)
            obj_match_set_th.append(obj_match)
            count_th += 1
        else:
            imgset_t.append(img_data)
            metaset_t.append(meta_mixed)
            labels_t.append(labels)
            z_set_t.append(z)
            obj_match_set_t.append(obj_match)
            count_t += 1

    print(f'needle-th count for {designated_class}: {count_th}')
    print(f'needle-t count for {designated_class}: {count_t}')

    imgset_th = np.array(imgset_th)
    metaset_th = np.array(metaset_th)
    labels_th = np.array(labels_th)
    z_set_th = np.array(z_set_th)
    obj_match_set_th = np.array(obj_match_set_th)

    imgset_t = np.array(imgset_t)
    metaset_t = np.array(metaset_t)
    labels_t = np.array(labels_t)
    z_set_t = np.array(z_set_t)
    obj_match_set_t = np.array(obj_match_set_t)

    if not os.path.exists(NEEDLE_SET_PATH):
        os.makedirs(NEEDLE_SET_PATH)

    NEEDLE_TH_SET_PATH = os.path.join(NEEDLE_SET_PATH, 'hosted_set')
    NEEDLE_T_SET_PATH = os.path.join(NEEDLE_SET_PATH, 'hostless_set')
    os.makedirs(NEEDLE_TH_SET_PATH, exist_ok=True)
    os.makedirs(NEEDLE_T_SET_PATH, exist_ok=True)

    NEEDLE_TH_LABEL_SET_PATH = os.path.join(NEEDLE_TH_SET_PATH, designated_class)
    NEEDLE_T_LABEL_SET_PATH = os.path.join(NEEDLE_T_SET_PATH, designated_class)
    os.makedirs(NEEDLE_TH_LABEL_SET_PATH, exist_ok=True)
    os.makedirs(NEEDLE_T_LABEL_SET_PATH, exist_ok=True)

    data_th_dict = {
        'imgset': imgset_th,
        'metaset': metaset_th,
        'labels': labels_th,
        'z_set': z_set_th,
        'obj_match_set': obj_match_set_th
    }

    data_t_dict = {
        'imgset': imgset_t,
        'metaset': metaset_t,
        'labels': labels_t,
        'z_set': z_set_t,
        'obj_match_set': obj_match_set_t
    }
    
    np.save(os.path.join(NEEDLE_TH_LABEL_SET_PATH, f'data_dict_{output_name}.npy'), data_th_dict)
    np.save(os.path.join(NEEDLE_T_LABEL_SET_PATH, f'data_dict_{output_name}.npy'), data_t_dict)




if __name__ == '__main__':
    designated_class = 'untouched'  # Can be changed

    if designated_class == 'SN':
        obj_list = load_samples(designated_class)

        worker = partial(process_obj_SN, obj_list=obj_list, designated_class=designated_class)
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(worker, list(range(len(obj_list))))

    elif designated_class == 'TDE' or designated_class == 'SLSN-I':
        obj_list = load_samples(designated_class)
        num_samples = 10000
        print('num_samples:', num_samples)
      
        worker = partial(process_obj_crossmatched, designated_class=designated_class)

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(worker, list(range(num_samples)))

    elif designated_class == 'untouched':
        untouched_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/NEEDLE2.0/untouched_2025/20240225_20250603.csv'
        obj_list = pd.read_csv(untouched_path)
        obj_list = set(obj_list['ZTFID'].tolist())

        print('obj_list:', len(obj_list))
        worker = partial(process_obj_untouched)

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(worker, obj_list)
    else:
        print('Invalid designated class')
        exit()

    save_results(results, designated_class, output_name = 'original_mask_2025')

   