import sys, os
import re
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from quality_classification_tf.quality_classification import QualityClassification
from image.image_restoration import ImageRestoration
from image.masking import Masking
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import stats
import tensorflow as tf
import warnings
from astropy.utils.exceptions import AstropyWarning
from config import DEFAULT_DATA_PATH, IMG_OUTPUT_PATH
import multiprocessing as mp
from scipy.ndimage import rotate
from astropy.stats import sigma_clip

from utils import load_sample_imgs, load_redshift_database, get_noise_distribution, display_image_pair
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


SCI_RE = re.compile('sci')
DIFF_RE = re.compile('diff')
REF_RE = re.compile('ref')
OBJ_RE = re.compile('ZTF')



def most_common(lst):
    return max(set(lst), key=lst.count)


class ImagePreprocessing:

    """
    This class is used to preprocess the image of the transient object.
    - img_path: str, path to the image data
    - host_data_path: str, path to the host data
    - mag_path: str, path to the magnitude data
    - output_path: str, path to save the output data
    - label_dict: dict, dictionary mapping labels
    """

    #--------------------------------
    # Initialization methods
    #--------------------------------
    def __init__(self, ztf_object=None, designated_class=None,
                 mag_path=None,
                 host_data_path=None,
                 img_path=None,
                 output_path=None,
                 ztf_obj_info_path=None,
                 load_img_path = None,
                 display = False,
                 augment = False,
                 train_mode = True,
                 masking = True):
        
        if ztf_object is None and designated_class is None:
            raise ValueError("Either ztf_object or designated_class must be provided")
        
        self.masking = masking
        self._display = display
        self.__band_dict = {'r': '2', 'g': '1'} # the band has priority but both bands should be considered. 

        self.__mag_path = os.path.join(DEFAULT_DATA_PATH, 'mag_sets_v4') \
            if mag_path is None else mag_path
        self.__host_data_path = os.path.join(DEFAULT_DATA_PATH, 'host_info_r5_ext') \
            if host_data_path is None else host_data_path
        self.__img_path = os.path.join(DEFAULT_DATA_PATH, 'image_sets_v3') \
            if img_path is None else img_path 
        self.__output_path = 'image_processing_output' \
            if output_path is None else output_path
        self.ztf_obj_info_path = ztf_obj_info_path

        if load_img_path is None:
            self.load_img_path = IMG_OUTPUT_PATH
        else:
            self.load_img_path = load_img_path

        self.img_sample_list = load_sample_imgs(self.load_img_path)

        if ztf_object is None and designated_class is not None:
            self.label_dict = self._load_class_samples(self.ztf_obj_info_path)
            self.ztf_object = self._shuffle_class_samples(designated_class)
        else:
            self.ztf_object = ztf_object

        print('image preprocessing ztf_object: ', self.ztf_object)
      
        self.__output_dir = os.path.join(self.__output_path, self.ztf_object)
        self.__mag_dir = os.path.join(self.__mag_path, self.ztf_object + '.json')
        self.__host_data_dir = os.path.join(self.__host_data_path, self.ztf_object + '.csv')
        self.__img_dir = os.path.join(self.__img_path, self.ztf_object)

        self.excuted = True 
        if not os.path.exists(self.__mag_dir):
            self.excuted = False
            print(f"Invalid magnitude directory: {self.__mag_dir}")
        
        if not os.path.exists(self.__img_dir):
            self.excuted = False
            print(f"Invalid image jsons directory: {self.__img_dir}")

        if self.img_sample_list is not None and self.ztf_object not in self.img_sample_list and train_mode:
            self.excuted = False
            print(f"Invalid image sample list: {self.ztf_object}")
        
        self.mag_data = self.load_mag_data
        self.excuted = False if self.mag_data is None else True
        
        self.host_data = self.load_host_data
        self.__quality_check_model = QualityClassification(verbose=False)
        
        try: 
            if self.excuted:
                if train_mode:
                    if self.img_sample_list is not None and self.ztf_object in self.img_sample_list:
                        self.img_data = self.load_obj_imgdata()
                        print(f"Loaded image data for {self.ztf_object}")

                    else:
                        self.meta = self._get_image_meta
                        self.mag_wg = self._get_image_with_mag
                        self.host_ra, self.host_dec = self._get_host_coords
                        self.target_ra, self.target_dec = self._get_target_coords
                        self.img_data = self.run_obj_plain(augment = augment)
                else:

                    # print('DEBUG: DESTINATION IMAGE PREPROCESSING, NO TRAIN MODE, SELF.MASKING: ', self.masking)
                    self.meta = self._get_image_meta
                    self.mag_wg = self._get_image_with_mag
                    self.host_ra, self.host_dec = self._get_host_coords
                    self.target_ra, self.target_dec = self._get_target_coords
                    self.img_data = self.run_obj_plain(augment = augment)
            else:         
        
                print('No image data available')
                self.img_data = None
        except:
            print('Excution fault, no image data available')
            self.img_data = None
                

    #--------------------------------
    # Class sample loading methods
    #--------------------------------
    def _load_class_samples(self, ztf_obj_info_path):
        

        if not os.path.exists(ztf_obj_info_path):
            raise FileNotFoundError(f"Label file not found: {ztf_obj_info_path}")
            
        info_df = pd.read_csv(ztf_obj_info_path)
        if info_df.empty:
            raise ValueError("Empty label file")
            
        if not all(col in info_df.columns for col in ['type', 'ZTFID']):
            raise ValueError("Label file missing required columns 'type' and/or 'ZTFID'")

        # only pick up the SLSN-I and TDE samples

        obj_dict = {'SLSN-I':info_df[info_df['type'] == 'SLSN-I']['ZTFID'].values.tolist(), 
               'TDE':info_df[info_df['type'] == 'TDE']['ZTFID'].values.tolist()}
        
        good_obj_dict = {'SLSN-I':[x for x in obj_dict['SLSN-I'] if x in self.img_sample_list], 
                         'TDE':[x for x in obj_dict['TDE'] if x in self.img_sample_list]}
        
        return good_obj_dict 


    def _shuffle_class_samples(self, label):
        """
        shuffle the class samples
        """

        if label not in self.label_dict:
            raise ValueError(f"Invalid label: {label}. Must be one of {list(self.label_dict.keys())}")

        if len(self.label_dict[label]) == 0:
            raise ValueError(f"No samples available for label: {label}")

        np.random.seed(None)
        return np.random.choice(self.label_dict[label], size=1, replace=False)[0]


    @property
    def img_redshift(self, input_z=None):
        """
        get the redshift of the object. If the spectroscopic redshift is not available, use the photometric redshift.
        """
        # try: 
        if input_z is None:
            obj_info = load_redshift_database()
            if self.ztf_object not in obj_info:
                return None
            archive_z = obj_info[self.ztf_object]
            if archive_z is not None:
                return archive_z
            else:
                if 'TNS' in self.mag_data and 'z' in self.mag_data['TNS'] and self.mag_data['TNS']['z'] is not None:
                    return float(self.mag_data['TNS']['z'])
                else:
                    return None       
        else:
            return input_z
        # except:
        #     print(f"Error loading image redshift for {self.ztf_object}")
        #     return None
    

    @property
    def load_host_data(self):
        '''
        load the host data
        '''
        if not os.path.exists(self.__host_data_dir):
            print(f"Host data file not found: {self.__host_data_dir}, this object might be hostless.")
            return None
        else:
            obj_host_data = pd.read_csv(self.__host_data_dir)
            if obj_host_data.empty:
                print(f"Host data file not found: {self.__host_data_dir}, this object might be hostless.")
                return None
            else:
                row = obj_host_data.iloc[0]
                host_data = row.to_dict()
                for k in ['Unnamed: 0', 'ra', 'dec']:
                    host_data.pop(k, None)
                try:
                    if self.mag_data is not None:
                        if 'sherlock' in self.mag_data and 'separationArcsec' in self.mag_data['sherlock']:
                            host_data['offset'] = float(self.mag_data['sherlock']['separationArcsec'])
                        else:
                            host_data['offset'] = None
                except:
                    host_data['offset'] = None
            
            return host_data
            

            
    @property
    def load_mag_data(self):
        """
        load the magnitude data
        """
        try: 
            with open(self.__mag_dir, 'r') as f:
                mag_data = json.load(f)
            return mag_data
        except:
            print(f"Error loading magnitude data for {self.ztf_object}")
            return None

    #--------------------------------
    # Image processing methods
    #--------------------------------
    def _zscale(self, img, log_img=False):
        """
        Apply Z-scale normalization to an image.

        Parameters:
        - img: ndarray, input image
        - log_img: bool, whether to apply logarithmic scaling

        Returns:
        - ndarray, normalized image
        """
        # vmin = visualization.ZScaleInterval().get_limits(img)[0]
        _, median, _ = stats.sigma_clipped_stats(img, mask=None, sigma=3.0, cenfunc='median')
        img = np.nan_to_num(img, nan=median)
        return np.log(img) if log_img else img

    def _image_normal(self, imgs):
        """
        Normalize the image data to the range [0, 1].

        Parameters:
        - imgs: list or numpy array, input images

        Returns:
        - list, normalized images
        """
        if not isinstance(imgs, (list, np.ndarray)):
            raise TypeError("imgs must be a list or numpy array")
            
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
            
        imgs_array = np.array(imgs)
        vmin = np.nanmin(imgs_array.flatten())
        vmax = np.nanmax(imgs_array.flatten())
        
        return [(img - vmin) / (vmax - vmin + 1e-6) for img in imgs]

    def _check_shape(self, img):
        """
        Check if the image has the shape (60, 60) and does not contain only NaN values.
        """
        return img is not None and img.shape == (60, 60) and not np.all(np.isnan(img))

    def _cutout_img_wcs(self, data, header, size=60):
        """
        Cutout the image to the proposed size.
        """
        pixels = WCS(header=header).all_world2pix(self.target_ra, self.target_dec, 1)
        pixels = [int(x) for x in pixels]
        cutout = Cutout2D(data, position=pixels, size=size)
        return cutout.data

    def plot_imgs(self, imgs):
        """
        show the test image
        """
        if len(imgs.shape) == 3:
            plt.figure(figsize=(imgs.shape[0]*3, 3))
            for i in range(imgs.shape[0]):
                plt.subplot(1, imgs.shape[0], i+1)
                plt.imshow(imgs[i])
                plt.title('score: {:.2f}'.format(self.quality_check(imgs[i].copy())))
                plt.gca().set_axis_off()
            plt.show()
        else:
            plt.figure(figsize=(3, 3))
            plt.imshow(imgs)
            plt.title('score: {:.2f}'.format(self.quality_check(imgs.copy())))
            plt.gca().set_axis_off()
            plt.show()

    def _cutout_img_pixel(self, img, size=60):
        """
        cut the image to the desired size
        """
        for i in range(2):
            if img.shape[i] > size:
                img = img[:size, :] if i == 0 else img[:, :size]
        return img

    #--------------------------------
    # Image quality check methods
    #--------------------------------
    def quality_check(self, image, threshold=0.5):
        """
        check the quality of the image
        """
        image_copy = image.copy()
        if np.isnan(image_copy).any():
            return False
        if self._check_shape(image_copy):
            result = self.__quality_check_model.run(image_copy)
            if result >= threshold:
                return True
            else:
                return False
        else: 
            return False
            

    def _get_header_data(self, filename):
        """
        Cutout image with 60x60 size and handle FITS file with different extensions.
        """
        # try:
        with fits.open(filename, ignore_missing_end=True) as f:
            if filename.endswith('fz'):
                f.verify('fix')
                # hdr = f[1].header
                # data = f[1].data
                hdr = f[0].header
                data = f[0].data
            else:
                hdr = f[0].header
                data = f[0].data
            self.fits_header = hdr
            if hdr['NAXIS1'] >= 60 and hdr['NAXIS2'] >= 60:
                data = self._cutout_img_pixel(data)
            if self._display:
                self.plot_imgs(data)
            if np.all(np.isnan(data)):
                return None, None
            else:
                return data, hdr
            
        # except Exception as e:
        #     print(f"Error processing file {filename}: {e}")
        #     return None, None


    #--------------------------------
    # Image coordinate mapping methods
    #--------------------------------

    def _check_sci_ref_alignment(self, sci_hdr, ref_hdr): 
        '''
        check if the science and reference images are aligned
        '''
        sci_ra, sci_dec = sci_hdr['CRVAL1'], sci_hdr['CRVAL2']
        ref_ra, ref_dec = ref_hdr['CRVAL1'], ref_hdr['CRVAL2']
        if abs(sci_ra - ref_ra) < 2 and abs(sci_dec - ref_dec) < 2:
            return True
        else:
            print(f"sci_ra: {sci_ra}, ref_ra: {ref_ra}, \n sci_dec: {sci_dec}, ref_dec: {ref_dec}")
            return False
        


    #--------------------------------
    # Image data loading methods
    #--------------------------------


    def _get_detection_with_mag(self, meta_path):
        m = open(meta_path, 'r')
        jfile = json.loads(m.read())
        return jfile["f2"]["withMag"]

    @property
    def _get_image_meta(self):
        with open(os.path.join(self.__img_dir, 'image_meta.json'), 'r') as mj:
            meta = json.load(mj)
        return meta
    
    @property
    def _get_image_with_mag(self):
        with open(os.path.join(self.__img_dir, 'mag_with_img.json'), 'r') as j:
            mag_wg = json.load(j)
        return mag_wg
    
    @property
    def _get_image_info(self): 
        '''
        get all the image information for g and r band.
        sort the image by the magnitude, from low to high. To make sure the peak data is the top one.
        '''
        band_data = {}
        for band in self.__band_dict:
            candids = self.mag_wg["candidates_with_image"][f'f{self.__band_dict[band]}']
            if not candids: #or self.meta[f'f{self.__band_dict[band]}']["obj_with_no_ref"]
                band_data[band] = None
            elif len(candids) == 0:
                band_data[band] = None
            else: # make sure there are mag data to process, otherwise return None
                mags = [[m['magpsf'], m["filefracday"]] for m in candids]
                mags.sort(key=lambda x: x[0])
                band_data[band] = np.array(mags)

        return band_data

    def _get_reference_image(self, band):
        """
        if one obj has multiple ref_imgs, and top one is bad, the other is good, we could still use it.
        """

        band_listdir = os.listdir(f"{self.__img_dir}/{self.__band_dict[band]}")
        ref_imgs = list(filter(REF_RE.match, band_listdir))
        
        good_ref, good_hdr = None, None
        bad_ref, bad_hdr = None, None

        if len(ref_imgs) >= 1:
            for fi in ref_imgs:
                data, hdr = self._get_header_data(f"{self.__img_dir}/{self.__band_dict[band]}/{fi}")

                if data is not None:
                    quality_score = self.quality_check(data)
                    if quality_score:
                        good_ref, good_hdr = data, hdr
                        # print(f"good ref: {self.__img_dir}/{self.__band_dict[band]}/{fi}")
                        # print('mean of good ref: ', np.nanmean(data))
                        break
                    else:
                        bad_ref, bad_hdr = data, hdr 
                        # print(f"bad ref: {self.__img_dir}/{self.__band_dict[band]}/{fi}")
                        # print('shape of bad ref: ', data.shape)
                        # print('mean of bad ref: ', np.nanmean(data))
                        continue
                else:
                    continue
        else:
            print(f'No reference image found for {band}-band')
       
        return good_ref, good_hdr, bad_ref, bad_hdr
    
    def _get_science_by_filefracday(self, filefracday, band): 
        '''
        get the science image and the header data
        do not check the quality score here as the bad images will be restored later.
        '''
        # try:
        obs_listdir = os.listdir(f"{self.__img_dir}/{self.__band_dict[band]}/{filefracday}")
        sci_img = list(filter(SCI_RE.match, obs_listdir))[0]
        sci_fitz = f"{self.__img_dir}/{self.__band_dict[band]}/{filefracday}/{sci_img}" 
        sci_data, hdr = self._get_header_data(sci_fitz)
       
        return sci_data, hdr
    
           
    def load_image_data(self):
        """
        Load image data for testing purposes.

        Returns:
            tuple: (sci_data, sci_hdr, ref_data, ref_hdr, image_flags)
                Contains image arrays, headers, and quality flags:
                1 = good, 0 = none, -1 = bad
        """

        band_data = self._get_image_info


        if band_data is None:
            print("No band data available")
            return None, None, None, None, {'science': 0, 'reference': 0}

        cache = None
        image_flags = {'science': 0, 'reference': 0}

        has_sci, has_ref = False, False

        for band in self.__band_dict:
            if band_data.get(band) is None:
                print(f'No {band}-band image data.')
                continue

            mags_info = band_data[band]
            if mags_info is None or len(mags_info) == 0:
                print(f'No magnitude information for {band}-band')
                continue

            ref_good, ref_hdr_good, ref_bad, ref_hdr_bad = self._get_reference_image(band)

            if ref_good is None and ref_bad is None:
                # print(f'No reference image data for {band}-band')
                continue

            sci_good, sci_hdr_good = None, None
            sci_bad, sci_hdr_bad = None, None

            # Try first 5 images by magnitude order
            for fracday in mags_info[:5, 1]:
                sci_data, sci_hdr = self._get_science_by_filefracday(fracday, band)
                if sci_data is None:
                    continue
                if self.quality_check(sci_data):
                    sci_good, sci_hdr_good = sci_data, sci_hdr
                    # print(f"good sci: {self.__img_dir}/{self.__band_dict[band]}/{fracday}")
                    break
                else:
                    sci_bad, sci_hdr_bad = sci_data, sci_hdr
                    # print(f"bad sci: {self.__img_dir}/{self.__band_dict[band]}/{fracday}")
    

    
            # If good pair found, return immediately
            if sci_good is not None and ref_good is not None:
                # print("Returning GOOD quality science and reference image data")
                return sci_good, sci_hdr_good, ref_good, ref_hdr_good, {'science': 1, 'reference': 1}
            
            # Cache this band's best data if no good match found
            has_sci = sci_good is not None or sci_bad is not None
            has_ref = ref_good is not None or ref_bad is not None

            if cache is None and has_sci and has_ref:
                cache = {
                    'sci_good': sci_good, 'sci_hdr_good': sci_hdr_good,
                    'sci_bad': sci_bad, 'sci_hdr_bad': sci_hdr_bad,
                    'ref_good': ref_good, 'ref_hdr_good': ref_hdr_good,
                    'ref_bad': ref_bad, 'ref_hdr_bad': ref_hdr_bad
                }

        if has_sci and has_ref:
            sci_data = sci_good if sci_good is not None else sci_bad
            sci_hdr = sci_hdr_good if sci_hdr_good is not None else sci_hdr_bad
            ref_data = ref_good if ref_good is not None else ref_bad
            ref_hdr = ref_hdr_good if ref_hdr_good is not None else ref_hdr_bad
            image_flags['science'] = 1 if sci_good is not None else (-1 if sci_data is not None else 0)
            image_flags['reference'] = 1 if ref_good is not None else (-1 if ref_data is not None else 0)
            return sci_data, sci_hdr, ref_data, ref_hdr, image_flags
        else: # the second band is also not good, then use the first band's data if possible.
            if cache is not None:
                sci_data = cache['sci_good'] if cache['sci_good'] is not None else cache['sci_bad']
                sci_hdr = cache['sci_hdr_good'] if cache['sci_good'] is not None else cache['sci_hdr_bad']
                ref_data = cache['ref_good'] if cache['ref_good'] is not None else cache['ref_bad']
                ref_hdr = cache['ref_hdr_good'] if cache['ref_good'] is not None else cache['ref_hdr_bad']
                image_flags['science'] = 1 if cache['sci_good'] is not None else (-1 if sci_data is not None else 0)
                image_flags['reference'] = 1 if cache['ref_good'] is not None else (-1 if ref_data is not None else 0)

                # print(f"Returning fallback image data: science ({image_flags['science']}), reference ({image_flags['reference']})")
                return sci_data, sci_hdr, ref_data, ref_hdr, image_flags
            else:
                # print("Returning None as no good quality image data found")
                return None, None, None, None, image_flags


    @property
    def _get_host_coords(self): 
        if os.path.exists(self.__host_data_dir):
            host_data = pd.read_csv(self.__host_data_dir)
            host_ra = float(host_data['ra'].iloc[0])
            host_dec = float(host_data['dec'].iloc[0])
            del host_data
            return host_ra, host_dec
        else:
            return None, None

    @property
    def _get_target_coords(self):
        if os.path.exists(self.__mag_dir):
            m = open(self.__mag_dir, 'r')
            jfile = json.loads(m.read())
            return jfile["objectData"]['ramean'], jfile["objectData"]['decmean']
        else:
            return None, None


    #--------------------------------
    # Image restoration methods
    #--------------------------------

    def run_obj(self, display = False, augment = True):

        """
        Runs image preprocessing pipeline for a given object.
        - Loads science and reference images
        - Restores low-quality images (if possible)
        - Applies masking and random augmentation (flip/rotate)
        - Returns N processed samples as a numpy array

        Args:
            sample_num (int): number of augmented samples to generate.

        Returns:
            np.ndarray: array of processed image samples with shape 
                        (sample_num, 2, H, W) [science, reference]
        """

        
        processed_img_array = []

        best_sci_data, best_sci_hdr, best_ref_data, best_ref_hdr, image_flags = self.load_image_data() 

        if image_flags['science'] != 0 and image_flags['reference'] != 0:
            alignment = self._check_sci_ref_alignment(best_sci_hdr, best_ref_hdr)

            if alignment:
                img_restoration = ImageRestoration(obj_id = self.ztf_object, 
                                                    sci_data = best_sci_data, sci_hdr = best_sci_hdr, 
                                                    ref_data = best_ref_data, ref_hdr = best_ref_hdr, 
                                                    target_ra = self.target_ra, target_dec = self.target_dec, 
                                                    host_ra = self.host_ra, host_dec = self.host_dec)
                
                # Initialize restoration score
                restore_score = 0.0

                if image_flags['science'] < 1 or image_flags['reference'] < 1:
                    
                    if image_flags['science'] == 1 and image_flags['reference'] == -1:
                        restore_score = img_restoration._SSIM_restore(is_sci = False , threshold = 0.2)
                        if display:
                            if img_restoration.sci_data is not None and img_restoration.ref_data is not None: 
                                display_image_pair(img_restoration.sci_data, img_restoration.ref_data, titles=None)
                            
                    
                    elif image_flags['science'] == -1 and image_flags['reference'] == 1:
                        restore_score = img_restoration._SSIM_restore(is_sci = True , threshold = 0.2)
                        if display:
                            if img_restoration.sci_data is not None and img_restoration.ref_data is not None: 
                                display_image_pair(img_restoration.sci_data, img_restoration.ref_data, titles=None)
            
                    else:
                        # print('The science and reference images are both bad quality.')
                        restore_score = 0.0
                    
                else:
                    restore_score = 1.0

                if restore_score >= 0.8:
                    if self.masking:
                        img_masking = Masking(sci_data = img_restoration.sci_data, ref_data = img_restoration.ref_data, 
                                            pixel_target = img_restoration.pixel_coords_target, 
                                            pixel_host = img_restoration.pixel_coords_host,
                                            display = self._display)
                        
                        img_masking._get_masked_img(sigma = 2)
                    
                        if augment:
                            if np.random.randint(2) == 0:
                                final_sci, final_ref = img_masking._flip_image()
                            else:
                                final_sci, final_ref = img_masking._rotate_image()
                        else:
                            final_sci, final_ref = img_masking.masked_sci_data, img_masking.masked_ref_data
                    else: # no masking, raw data
                        final_sci, final_ref = img_restoration.sci_data, img_restoration.ref_data

                    processed_img_array = np.array(self._image_normal([final_sci, final_ref]))

        return processed_img_array

    def run_obj_plain(self, display = False, augment = False):
        '''
        abandon this function. 
        '''
        processed_img_array = None

        best_sci_data, best_sci_hdr, best_ref_data, best_ref_hdr, image_flags = self.load_image_data() 

        if image_flags['science'] != 0 and image_flags['reference'] != 0:
            alignment = self._check_sci_ref_alignment(best_sci_hdr, best_ref_hdr)
  
            if alignment:
          
                img_restoration = ImageRestoration(obj_id = self.ztf_object, 
                                                    sci_data = best_sci_data, sci_hdr = best_sci_hdr, 
                                                    ref_data = best_ref_data, ref_hdr = best_ref_hdr, 
                                                    target_ra = self.target_ra, target_dec = self.target_dec, 
                                                    host_ra = self.host_ra, host_dec = self.host_dec)
                
                # Initialize restoration score
                restore_score = 0.0

                if image_flags['science'] < 1 or image_flags['reference'] < 1:
                    
                    if image_flags['science'] == 1 and image_flags['reference'] == -1:
                        restore_score = img_restoration._SSIM_restore(is_sci = False , threshold = 0.2)  
                    elif image_flags['science'] == -1 and image_flags['reference'] == 1:
                        restore_score = img_restoration._SSIM_restore(is_sci = True , threshold = 0.2)
                    else:
                        # print('The science and reference images are both bad quality.')
                        restore_score = 0.0
                    
                else:
                    restore_score = 1.0

                if restore_score >= 0.8:
                    
                    if self.masking:
                        img_masking = Masking(sci_data = img_restoration.sci_data, ref_data = img_restoration.ref_data, 
                                        pixel_target = img_restoration.pixel_coords_target, 
                                        pixel_host = img_restoration.pixel_coords_host,
                                        display = display)
                        img_masking._get_masked_img(sigma = 2)
                        if augment:
                            if np.random.randint(2) == 0:
                                final_sci, final_ref = img_masking._flip_image()
                            else:
                                final_sci, final_ref = img_masking._rotate_image()
                        else:
                            final_sci, final_ref = img_masking.masked_sci_data, img_masking.masked_ref_data
                    else:
                        print('DEBUG: DESTINATION IMAGE PREPROCESSING, NO MASKING, RAW DATA')
                        final_sci, final_ref = img_restoration.sci_data, img_restoration.ref_data

          

                    

                    processed_img_array = np.array(self._image_normal([final_sci, final_ref]))
                    
             

        return processed_img_array       
    
    #--------------------------------
    # Save image data methods
    #--------------------------------

    def save_obj_imgdata(self):
        '''
        save the science and reference image data to a file
        '''
        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir, exist_ok=True)
            img_data = self.run_obj_plain()
            if img_data is not None:
                np.save(self.__output_dir + '/imgdata.npy', img_data)
                print(f'Image data saved to {self.__output_dir}/imgdata.npy')
            else:
                print('No good image data to save.')
        else:
            print(f'Image data already exists.')
        
    def load_obj_imgdata(self):
        '''
        load the science and reference image data from a file
        '''
        if os.path.exists(self.load_img_path + f'/{self.ztf_object}/imgdata.npy'):
            img_data = np.load(self.load_img_path + f'/{self.ztf_object}/imgdata.npy')
            self.ztf_object = self.ztf_object
            return img_data
        else:
            print(f'No image data found on {self.load_img_path}/{self.ztf_object}/imgdata.npy')
            return None
    
    def get_needle_imgdata(self, sample_num = 1): 
        '''
        get the science and reference image data for NEEDLE inputs.
        '''
        processed_img_array = self.run_obj(sample_num = sample_num)

        return processed_img_array
        
    def augment_imgdata(self, imgs):
        '''
        augment the image data
        '''
        print('--- augmenting image data ---')
        np.random.seed()
        if imgs[0] is not None and imgs[1] is not None and self.masking:
            if np.random.randint(2) == 0:
                print('flipping image')
                return self.flip_image(imgs)
            else: 
                print('rotating image')
                return self.rotate_image(imgs)
        else:
            print('No masked image found or masking is disabled')
            return imgs
        

    def flip_image(self, imgs, axis = None):
        '''
        This function is to flip the image randomly
        '''
        np.random.seed()
        if axis is None:
            axis = np.random.randint(2)
        flipped_sci = None
        flipped_ref = None
        if imgs[0] is not None and imgs[1] is not None:
            flipped_sci = np.flip(imgs[0], axis=axis)
            flipped_ref = np.flip(imgs[1], axis=axis)
        else:
            print('No masked image found')
        

        return flipped_sci, flipped_ref



    def rotate_image(self, imgs, angle = None):
        """
        Rotate a square image by a given angle and fill missing edges with sampled noise.
        
        Parameters:
            image (np.ndarray): 2D square image array.
            angle (float): Rotation angle in degrees (counter-clockwise).
           
        Returns:
            np.ndarray: Rotated image with noise-filled edges.
        """
        
        def check_bound(image): 
            clipped = sigma_clip(image, sigma=3, maxiters=3)
            mask = clipped.mask
     
            flag = False
            if np.any(mask[:,0]) or np.any(mask[:, -1]) or np.any(mask[0, :]) or np.any(mask[-1, :]):
                flag = True
            return flag
            
        
        def rotate_single_image(image):
            if image.ndim != 2 or image.shape[0] != image.shape[1]:
                print("Input image must be a square 2D array.")
                return None
        
            rotated = rotate(image, angle, reshape=False, order=1, mode='constant', cval=np.nan)
            noise_values = get_noise_distribution(image, factor1 = 3, factor2 = 1)
            noise = np.random.choice(noise_values, size=rotated.shape, replace=True)
            filled_rotated = np.where(np.isnan(rotated), noise, rotated)
            return filled_rotated

        np.random.seed()
        if angle is None:
            angle = np.random.randint(360)
        if check_bound(imgs[0]) and angle%90 != 0:
            angle = 90 * np.random.randint(4)
         
        print('rotation angle: ', angle)
        rotated_sci = rotate_single_image(imgs[0])
        rotated_ref = rotate_single_image(imgs[1])
   
        return [rotated_sci, rotated_ref]

 
    
    #--------------------------------
    # Plotting methods
    #--------------------------------


        
    def plot_mask(self, image, mask1, mask2, fname):
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')#, vmin=np.min(sci_data))
        plt.subplot(1, 3, 2)
        plt.imshow(mask1, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(mask2, cmap='gray')
        plt.savefig(f'{self.__output_dir}/{fname}.png')

    def plot_restore(self, image, image1):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')#, vmin=np.min(sci_data))
        plt.subplot(1, 2, 2)
        plt.imshow(image1, cmap='gray')
        plt.savefig(f'{self.__output_dir}/restored.png')

    def plot_coordinates(self, image, target_coords, host_coords):
        plt.imshow(image)
        plt.plot(target_coords[0], target_coords[1], marker='o', color='red', markersize=5, label= 'target RA/DEC')
        plt.plot(host_coords[0], host_coords[1], marker='o', color='blue', markersize=5, label= 'host RA/DEC')
        plt.legend()
        plt.savefig(f'{self.__output_dir}/coords.png')

    def plot_img(self, image, fname):
        plt.clf()
        plt.imshow(image, cmap='gray')#, vmin=np.min(sci_data))
        plt.savefig(f'{self.__output_dir}/{fname}.png')


def process_obj(obj, mag_path, host_path, img_path, output_path):
    if not os.path.exists(output_path + obj):
        img_demo = ImagePreprocessing(ztf_object = obj, 
                                        mag_path = mag_path,
                                        host_data_path = host_path,
                                        img_path = img_path,
                                        output_path = output_path,
                                        ztf_obj_info_path = None,
                                        display = False,
                                        augment = False,
                                        train_mode = False,
                                        masking = False) 
        img_demo.save_obj_imgdata()
        print(f'Successfully processed {obj}')
    else:
        print(f'object {obj} already exists.')
        

def process_obj_wrapper(args):
    '''
    Wrapper function for multiprocessing
    '''
    obj, mag_path, host_path, img_path, output_path = args
    return process_obj(obj, mag_path, host_path, img_path, output_path)
# untouched_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/NEEDLE2.0/untouched_2024/'

# def process_obj(obj):
#     if not os.path.exists(os.path.join(untouched_path, 'image_preprocessing_output', obj)):
#         img_demo = ImagePreprocessing(ztf_object = obj, 
#                                         mag_path=os.path.join(untouched_path, 'mags'),
#                                         host_data_path=os.path.join(untouched_path, 'hosts_ext'),
#                                         img_path=os.path.join(untouched_path, 'images'),
#                                         output_path=os.path.join(untouched_path, 'image_preprocessing_output'),
#                                         ztf_obj_label_path=None,
#                                         display = False,
#                                         augment = False)
#         img_demo.save_obj_imgdata()
#     else:
#         print(f'{obj} already processed')

#     # In one cell:


# In another cell:
if __name__ == '__main__':
    # ztf_obj_info_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/NEEDLE2.0/info/ztf_train_valid_set.csv'
    # info_df = pd.read_csv(ztf_obj_info_path)
    # # Remove duplicates by ZTFID
    # info_df = info_df.drop_duplicates(subset=['ZTFID'])
    # obj_list = info_df['ZTFID'].values.tolist()

    # img_path = '/Users/xinyuesheng/Documents/astro_projects/data/image_sets_v3'
    # mag_path = '/Users/xinyuesheng/Documents/astro_projects/data/mag_sets_v4'
    # host_path = '/Users/xinyuesheng/Documents/astro_projects/data/host_info_r5_ext'
    # output_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/NEEDLE2.0/image/image_unmasked_output/'


    
    # with mp.Pool(processes=8) as pool:
    #     pool.map(process_obj_wrapper, [(obj, mag_path, host_path, img_path, output_path) for obj in obj_list])




    untouched_path = '/Users/xinyuesheng/Documents/astro_projects/scripts/NEEDLE2.0/untouched_2025/'
    obj_list = pd.read_csv(untouched_path + '20240225_20250603.csv')
    obj_list = obj_list['ZTFID'].values.tolist()
    mag_path = untouched_path + 'mags/'
    host_path = untouched_path + 'hosts_ext/'
    img_path = untouched_path + 'images/'
    output_path = untouched_path + 'image_unmasked_output/'

    with mp.Pool(processes=8) as pool:
        pool.map(process_obj_wrapper, [(obj, mag_path, host_path, img_path, output_path) for obj in obj_list])


    # for obj in obj_list:
    #     img_demo = ImagePreprocessing(ztf_object = obj, 
    #                                 mag_path=os.path.join(untouched_path, 'mags'),
    #                                 host_data_path=os.path.join(untouched_path, 'hosts_ext'),
    #                                 img_path=os.path.join(untouched_path, 'images'),
    #                                 output_path=os.path.join(untouched_path, 'image_unmasked_output'),
    #                                 ztf_obj_label_path=None,
    #                                 display = False,
    #                                 augment = False,
    #                                 train_mode = False,
    #                                 masking = False)
    #     img_demo.save_obj_imgdata()

