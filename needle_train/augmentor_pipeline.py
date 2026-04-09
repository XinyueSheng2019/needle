import numpy as np
import pandas as pd
from light_curve.light_curve_upsampling import LightCurveUpsamplingPipeline
from image.image_preprocessing import ImagePreprocessing
from quality_classification_tf.quality_classification import QualityClassification
from utils import load_sample_imgs, load_sample_lc, load_redshift_database
from config import *
import matplotlib.pyplot as plt
import os




class DataAugmentor:

    def __init__(self, designated_class = 'TDE', obj_A = None, obj_B = None, img_list = None, lc_list = None, display = False, output_path = 'new_samples_plots', augment = False, train_mode = True, masking = True, load_img_path = None):

        self.designated_class = designated_class
        self.excuted = True
        self._augment = augment
        self.quality_check = QualityClassification(verbose=False)
        self._display = display
        self.obj_A = obj_A
        self.obj_B = obj_B   
        self.output_path = output_path
        self.img_list = img_list
        self.lc_list = lc_list
        self.objs_z = load_redshift_database()
        self.objs_z_converted = self.converted_redshift_dict()
        self.train_mode = train_mode
        self.masking = masking
        self.load_img_path = load_img_path
        if self.load_img_path is None:
            if self.masking:
                self.load_img_path = IMG_OUTPUT_PATH
            else:
                self.load_img_path = UNMASKED_IMG_OUTPUT_PATH

    
    def converted_redshift_dict(self):
        obj_z_converted = {}
        if self.objs_z is not None:
            for obj_id, z in self.objs_z.items():
                float_z = float(z)
                if float_z not in obj_z_converted:
                    obj_z_converted[float_z] = [obj_id]
                else:
                    obj_z_converted[float_z].append(obj_id)
      
            return obj_z_converted
        else:
            print('No redshift dictionary available')
            return None
        
    def naive_run(self, ztf_id, mag_path = None, host_data_path = None, img_path = None):
        '''for SN dataset that does not need cross-matching or image augmentation'''
        run_success = True
        meta_r, meta_mixed, img_data, lc_data =  None, None, None, None
        if self.img_list is None:
            self.img_list = load_sample_imgs()
        if self.lc_list is None:
            self.lc_list = load_sample_lc()
    
        if self.lc_list is not None and self.img_list is not None:
            self.obj_A = ztf_id
            self.obj_B = ztf_id
         
            self.image_preprocessing = ImagePreprocessing(ztf_object = self.obj_A, 
                                                          mag_path = mag_path,
                                                          host_data_path = host_data_path,
                                                          img_path = img_path,
                                                          load_img_path = self.load_img_path,
                                                          display = self._display,
                                                          augment = self._augment,
                                                          train_mode = self.train_mode,
                                                          masking = self.masking)
            img_data = self.image_preprocessing.img_data
            
            if img_data is not None:
                print('####################### img_data loaded successfully')
                self.photo_upsampling = LightCurveUpsamplingPipeline(ztf_object = self.obj_B, 
                                                                     mag_path= mag_path,
                                                                     img_host_data = self.image_preprocessing.host_data, 
                                                                     img_z = self.image_preprocessing.img_redshift,
                                                                     gp_fitting = False,
                                                                     min_detection = 1) # loose rule for SN light curves as they do not need to upsample
                if self.photo_upsampling.executed:
                    lc_data = self.photo_upsampling.lc_data
                    print('#################### lc_data loaded successfully')
                    if lc_data is not None:
                        meta_r, meta_mixed, find_host = self.photo_upsampling.get_needle_meta(upsampled_lc = lc_data)
                        print('#################### meta_r, meta_mixed, find_host loaded successfully')
                    else:
                        print('No light curve data available or out of ZTF limit')
                        run_success = False
                else:
                    print('No light curve data available or out of ZTF limit')
                    run_success = False
            else:
                print('No image data available')
                run_success = False
        else:
            print('No light curve or image data sets available')
            run_success = False

        if run_success:
            return img_data, lc_data, meta_r, meta_mixed, find_host
        else:
            return None, None, None, None, None

    

    def next_run_fast(self, init = True):
        '''for SLSN and TDE cross-matching datasets'''
        run_success = True
        meta_r, meta_mixed, img_data, upsampled_lc = None, None, None, None
        if init:
            self.obj_A = None
            self.obj_B = None

        if self.lc_list is not None and self.img_list is not None:
            check = self.shuffle_pairs()
            if not check:
                print('No object in redshift range')
                return None, None, None, None, None
            
            self.image_preprocessing = ImagePreprocessing(ztf_object = self.obj_A, 
                                                          display = self._display,
                                                          augment = self._augment,
                                                          train_mode = self.train_mode,
                                                          masking = self.masking)
            img_data = self.image_preprocessing.img_data
            if img_data is not None:
                img_data = self.image_preprocessing.augment_imgdata(img_data)
                self.photo_upsampling = LightCurveUpsamplingPipeline(ztf_object = self.obj_B, 
                                                                     img_host_data = self.image_preprocessing.host_data, 
                                                                     img_z = self.image_preprocessing.img_redshift,
                                                                     gp_fitting = True,
                                                                     load_gp = True,
                                                                     min_detection = 1)
                if self.photo_upsampling.executed:
                    upsampled_lc = self.photo_upsampling.upsample_light_curve()
                    if upsampled_lc is not None:
                        meta_r, meta_mixed, find_host = self.photo_upsampling.get_needle_meta(upsampled_lc)
                    else:
                        print('No light curve data available or out of ZTF limit')
                        run_success = False
                else:
                    print('No light curve data available or out of ZTF limit')
                    run_success = False
                    
            else:
                print('No image data available')
                run_success = False

            if run_success:
                print('run successfully')

                if self._display:
                    self.image_preprocessing.plot_imgs(img_data)
                    self.photo_upsampling.plot_light_curves(upsampled_lc)

                return img_data, upsampled_lc, meta_r, meta_mixed, find_host
            else:
                print('No light curve or image data sets available')
                return None, None, None, None, None
        else:
            print('No light curve or image data sets available')
            return None, None, None, None, None
       

    def next_run(self, mag_path = None, host_data_path = None, img_path = None, save = True):
        try: 
            self.obj_A = None
            self.obj_B = None
            check = self.shuffle_pairs()
            if not check:
                print('No object in redshift range')
                return None, None, None
            
            self.image_preprocessing = ImagePreprocessing(ztf_object = self.obj_A, 
                                                          mag_path= mag_path,
                                                          host_data_path= host_data_path,
                                                          img_path= img_path,
                                                          load_img_path = self.load_img_path,
                                                          display = self._display,
                                                          augment = self._augment,
                                                          train_mode = self.train_mode,
                                                          masking = self.masking)
            self.img_z = self.image_preprocessing.img_redshift
            self.photo_upsampling = LightCurveUpsamplingPipeline(ztf_object = self.obj_B, img_host_data = self.image_preprocessing.host_data)
            self.lc_z = self.photo_upsampling.lc_redshift
            processed_img_A, host_A = self.get_image_A()
            converse_lc_B = self.get_light_curve_B()
            if converse_lc_B is not None:
                self.meta_r, self.meta_mixed, self.find_host = self.photo_upsampling.get_needle_meta(upsampled_lc = converse_lc_B)
            else:
                self.meta_r, self.meta_mixed, self.find_host = None, None, None
            if self._display:
                self.image_preprocessing.plot_imgs(processed_img_A[0])
                self.photo_upsampling.plot_light_curves(converse_lc_B)  
            if save:   
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path)
                self.save_as_png(self.photo_upsampling.lc_data, converse_lc_B, processed_img_A[0], output_path = self.output_path)
            return processed_img_A, host_A, converse_lc_B
        except:
            return None, None, None


    def get_obj_in_redshift_range(self, current_z, threshold = None, fixed_range = None):
        if self.objs_z_converted is None:
            print('No redshift dictionary available')
            return None
        else:
            if threshold is not None: 
                z_range = [current_z - current_z * threshold, current_z + current_z * threshold]
                print('test z_range: ', z_range)
                partial_objs = []
                for z in self.objs_z_converted.keys():
                    if z >= z_range[0] and z <= z_range[1]:
                        partial_objs += self.objs_z_converted[z]
                if len(partial_objs) == 0:
                    print('No object in redshift range')
                    return None
                return partial_objs
            elif fixed_range is not None:
                z_range = [current_z - fixed_range, current_z + fixed_range]
                partial_objs = []
                for z in self.objs_z_converted.keys():
                    if z >= z_range[0] and z <= z_range[1]:
                        partial_objs += self.objs_z_converted[z]
                if len(partial_objs) == 0:
                    print('No object in redshift range')
                    return None
                return partial_objs
            else:
                print('threshold and fixed_range are both None')
                return None

    def shuffle_pairs(self):
        # shuffle the pairs of image and light curve with the same redshift range
        np.random.seed()
        if self.objs_z is None:
            print('No redshift dictionary available')
            return False 
        self.obj_list = self.load_samples(self.designated_class)
        if self.img_list is None:
            self.img_list = load_sample_imgs()
            good_img_ids = [x for x in self.obj_list if x in self.img_list]
        else:
            good_img_ids = self.img_list
        if self.lc_list is None:
            self.lc_list = load_sample_lc()
            good_lc_ids = [x for x in self.obj_list if x in self.lc_list]
        else:
            good_lc_ids = self.lc_list
        
        # good_img_ids = [x for x in self.obj_list if x in self.img_list]
        self.obj_A = np.random.choice(good_img_ids, size = 1)[0]
        if self.obj_A not in self.objs_z:
            print('No redshift for object A in redshift database')
            return False
        self.obj_A_z = self.objs_z[self.obj_A]
        obj_B_list = self.get_obj_in_redshift_range(self.obj_A_z, threshold = 0.2)
            # print('test obj_B_list: ', obj_B_list)
        if obj_B_list is not None:
            good_obj_B_list = [x for x in obj_B_list if x in good_lc_ids]
            if len(good_obj_B_list) == 0:
                print('No object in redshift range')
                return False 
            self.obj_B = np.random.choice(good_obj_B_list, size = 1)[0]
        else:
            print('No object in redshift range')
            return False 

        print('obj_A: ', self.obj_A)
        print('obj_B: ', self.obj_B)
        return True 


    # --- get image data --- 
    def get_image_A(self):
        print('--- get image A ---')
        processed_img_A = self.image_preprocessing.run_obj(display = self._display, augment = self._augment)
        host_A = self.image_preprocessing.load_host_data
        return processed_img_A, host_A
    

    # --- get photometry data ---                  
    def get_light_curve_B(self):
        print('--- get light curve B ---')
        upsampled_lc = self.photo_upsampling.upsample_light_curve()
        if upsampled_lc is None:
            print('No light curve data available or out of ZTF ')
            return None
        print('--- apply img_z to lc ---')
        converse_lc = self.photo_upsampling.apply_img_z_to_lc(upsampled_lc)
        return converse_lc 


    # --- file management ---
    def load_samples(self, designated_class = 'TDE'):
        f = pd.read_csv(OBJ_INFO_PATH)
        if designated_class == 'SN':
            class_list = f[(f.type != 'TDE') & (f.type != 'SLSN-I')]['ZTFID'].tolist()
        else:
            class_list = f[f.type == designated_class]['ZTFID'].tolist()
        return list(set(class_list))


    def save_as_png(self, 
                    light_curve_B0,
                    light_curve_B, 
                    processed_img_A, 
                    output_path=None, 
                    img_shape=(60, 60), 
                    figsize=(12, 12), 
                    dpi=100, 
                    cmap='gray'):
        """
        Save light curve, images, and object info as a PNG.

        Layout:
        [ Light Curve   |  Info Text  ]
        [ Science Image | Reference Image ]
        """
        try:
            if light_curve_B0 is None or light_curve_B is None or processed_img_A is None:
                print("Error: One or more input data is None")
                return

            if not isinstance(processed_img_A, (list, np.ndarray)) or len(processed_img_A) < 2:
                print("Error: processed_img_A must be a list/array with at least 2 elements")
                return

            plt.figure(figsize=figsize, dpi=dpi)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.2, hspace=0.2)

            # Light Curve (Top Left)
            plt.subplot(2, 2, 1)
            plt.title(f'{self.obj_B} - Light Curve (z = {self.lc_z})')
            plt.xlabel('Time')
            plt.ylabel('Magnitude')
            if 'ztfg' in light_curve_B0['band'].values:
                plt.errorbar(light_curve_B0[light_curve_B0.band == 'ztfg']['time'],
                            light_curve_B0[light_curve_B0.band == 'ztfg']['mag'],
                            yerr=light_curve_B0[light_curve_B0.band == 'ztfg']['mag_err'],
                            fmt='o', color='green', label='ztf_g')
            if 'ztfr' in light_curve_B0['band'].values:
                plt.errorbar(light_curve_B0[light_curve_B0.band == 'ztfr']['time'],
                            light_curve_B0[light_curve_B0.band == 'ztfr']['mag'],
                            yerr=light_curve_B0[light_curve_B0.band == 'ztfr']['mag_err'],
                            fmt='o', color='red', label='ztf_r')
            plt.gca().invert_yaxis()
            plt.legend()

            # Info Text (Top Right)
            plt.subplot(2, 2, 2)
            plt.title(f'Converted Light Curve (z = {self.img_z})')
            plt.xlabel('Time')
            plt.ylabel('Magnitude')
            if 'ztfg' in light_curve_B['band'].values:
                plt.errorbar(light_curve_B[light_curve_B.band == 'ztfg']['time'],
                            light_curve_B[light_curve_B.band == 'ztfg']['mag'],
                            yerr=light_curve_B[light_curve_B.band == 'ztfg']['mag_err'],
                            fmt='o', color='green', label='ztf_g')
            if 'ztfr' in light_curve_B['band'].values:
                plt.errorbar(light_curve_B[light_curve_B.band == 'ztfr']['time'],
                            light_curve_B[light_curve_B.band == 'ztfr']['mag'],
                            yerr=light_curve_B[light_curve_B.band == 'ztfr']['mag_err'],
                            fmt='o', color='red', label='ztf_r')
            plt.gca().invert_yaxis()
            plt.legend()

            # Science Image (Bottom Left)
            plt.subplot(2, 2, 3)
            plt.title(f'{self.obj_A} - Science')
            sci_img = processed_img_A[0].reshape(*img_shape)
            vmin, vmax = np.nanmin(sci_img), np.nanmax(sci_img)
            plt.imshow(sci_img, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            plt.axis('off')

            # Reference Image (Bottom Right)
            plt.subplot(2, 2, 4)
            plt.title(f'{self.obj_A} - Reference')
            ref_img = processed_img_A[1].reshape(*img_shape)
            plt.imshow(ref_img, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            plt.axis('off')

            # Global figure title
            plt.suptitle(f'{self.designated_class} generated sample', fontsize=18, y=0.95)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save the figure
            if output_path is None:
                output_filename = f'{self.obj_A}_{self.obj_B}.png'
            else:
                os.makedirs(output_path, exist_ok=True)
                output_filename = os.path.join(output_path, f'{self.obj_A}_{self.obj_B}.png')

            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f'Successfully saved to {output_filename}')

        except Exception as e:
            print(f"Error in save_as_png: {str(e)}")
            plt.close()  # Ensure figure is closed even if error occurs






    
    
    


