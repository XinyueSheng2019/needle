import sys
import json
import numpy as np
import pandas as pd
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy.ndimage import binary_dilation, rotate, label
from sklearn.cluster import DBSCAN
from quality_classification_tf.quality_classification import QualityClassification
from utils import get_noise_distribution, show_images


# Define the 8 possible directions for movement
dx = [-1, 0, 1, 1, 1, 0, -1, -1]
dy = [1, 1, 1, 0, -1, -1, -1, 0]
noise_factor1 = 3
noise_factor2 = 1

class Masking:
    """
    This class is used to mask the image, and simulate the missing data in the image.
    """
    def __init__(self, sci_data, ref_data, pixel_target, pixel_host, display = False):
        self.sci_data = sci_data
        self.ref_data = ref_data

        self._update_pixel_coords(pixel_host, pixel_target)

        self.host_region = None
        self.target_region = None

        self.masked_sci_data = None
        self.masked_ref_data = None
        self.sci_mask = None
        self.ref_mask = None 
        self.mask = None
        self._display = display
        self.sci_noise = None
        self.ref_noise = None

        self.__fixed_center = self._get_fixed_pixels
        self.__scaling_img = [np.nanmin(self.sci_data), np.nanmax(self.sci_data)]
        self.quality_check = QualityClassification(verbose=False)
    

    def _update_pixel_coords(self, pixel_host, pixel_target):
        print('masking test: pixel_host: ', pixel_host)
        print('masking test: pixel_target: ', pixel_target)
        self.__pixel_coords_host = [int(pixel_host[1]), int(pixel_host[0])] if pixel_host is not None else None
        self.__pixel_coords_target = [int(pixel_target[1]), int(pixel_target[0])]
        self.__pixel_coords_host = self._bound_host_pixel(self.__pixel_coords_host)


    def _bound_host_pixel(self, coord_pixel):
        """Ensure pixel coordinates stay within 60x60 image bounds"""
        if coord_pixel is None:
            return None
        coord_pixel[0] = np.clip(coord_pixel[0], 0, 59)
        coord_pixel[1] = np.clip(coord_pixel[1], 0, 59)
        return coord_pixel
        
        
    @property
    def _get_fixed_pixels(self):
        # Set the fixed area to add to the mask
        temp_mask = np.zeros_like(self.sci_data)
        height, width = self.sci_data.shape

        def mark_region_around(coord, label="unknown", size = 3):
            y, x = coord  
            # Check if the coord is out of bounds
            if not (0 <= x < width and 0 <= y < height):
                print(f"[Warning] {label} coordinate {coord} is out of bounds for a {width}x{height} image.")
                return False  # signal invalid coord

            # Clip the region to image bounds
            x_start = max(x - size, 0)
            x_end = min(x + size, width)
            y_start = max(y - size, 0)
            y_end = min(y + size, height)

            if x_start == 0 or x_end == width or y_start == 0 or y_end == height:
                print(f"[Info] {label} coordinate {coord} is near the image border.")

            for j in range(y_start, y_end):  # y corresponds to rows
                for i in range(x_start, x_end):  # x corresponds to columns
                    temp_mask[j, i] = 1

            return True

        # Apply to target
        mark_region_around(self.__pixel_coords_target, label="Target")

        # Apply to host if available and valid
        if self.__pixel_coords_host is not None:
            mark_region_around(self.__pixel_coords_host, label="Host")


        return temp_mask.astype(bool)

    def _get_host_coords(self, host_path):
        host_data = pd.read_csv(host_path)
        host_ra, host_dec = host_data['ra'][0], host_data['dec'][0]
        del host_data
        return float(host_ra), float(host_dec)

    def _get_target_coords(self, target_path):
        m = open(target_path, 'r')
        jfile = json.loads(m.read())
        return jfile["objectData"]['ramean'], jfile["objectData"]['decmean']

    def _get_detection_with_mag(self, meta_path):
        m = open(meta_path, 'r')
        jfile = json.loads(m.read())
        return jfile["f2"]["withMag"]
    
    def _zscale(self, image, contrast=0.5, nsamples=1000):
        """
        Implement zscale using Astropy's ZScaleInterval
        
        Parameters:
        -----------
        image : ndarray
            Input image data
        contrast : float, optional
            The scaling contrast (default=0.25)
        nsamples : int, optional
            Number of samples to use for estimating the scaling (default=1000)
        
        Returns:
        --------
        vmin, vmax : tuple
            The suggested minimum and maximum values for display
        """
        zscale = ZScaleInterval(contrast=contrast, n_samples=nsamples)
        vmin, vmax = zscale.get_limits(image)
        image_scaled = ((image - vmin) / (vmax - vmin))

        show_images(image, image_scaled, global_scale = False, titles=['original image', 'zscaled image'])
        return image_scaled

    def _image_normal(self, img):
        """
        Normalize the image data to the range [0, 1].

        Parameters:
        - img: ndarray, input image

        Returns:
        - ndarray, normalized image
        """
        nor_img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
        return nor_img
    
    
    def _mask_nearby_sources(self, pixel_regions, is_sci = True):
        '''
        mask nearby sources.
        '''
        data = self.sci_data.copy() if is_sci else self.ref_data.copy()
        noise = self.sci_noise if is_sci else self.ref_noise

        if noise is None:
            noise = get_noise_distribution(data, factor1 = noise_factor1, factor2 = noise_factor2)

        data[pixel_regions] = np.random.choice(noise, size=np.sum(pixel_regions), replace=True)

        if is_sci:
            self.masked_sci_data = data
        else:
            self.masked_ref_data = data
        
     

    def _get_closest_cluster(self, target_coord, w, num_ccl):
        """
        Find the cluster closest to the target coordinate.
        
        Args:
            target_coord: Target coordinate (y, x)
            w: 2D array where each value represents a cluster ID
            num_ccl: Number of clusters
            
        Returns:
            2D array with 1s marking the closest cluster
        """
        # Initialize result array
        res_data = np.zeros(w.shape)
        
        # Skip if no clusters
        if num_ccl <= 1:
            return res_data
            
        # Calculate distances for each cluster
        min_dist = float('inf')
        closest_cluster = None
        
        for cc_idx in range(1, num_ccl+1):
            # Get coordinates of current cluster
            cluster_coords = np.argwhere(w == cc_idx)
            
            # Skip empty clusters
            if len(cluster_coords) == 0:
                continue
                
            # Calculate distances to target
            distances = np.linalg.norm(cluster_coords - target_coord, axis=1)
            min_cluster_dist = np.min(distances)
            
            # Update if this cluster is closer
            if min_cluster_dist < min_dist:
                min_dist = min_cluster_dist
                closest_cluster = cluster_coords
        
        # If no valid clusters found
        if closest_cluster is None:
            return res_data
            
        # Mark the closest cluster
        for coord in closest_cluster:
            res_data[coord[0], coord[1]] = 1
            
        return res_data
    
    def _spatial_cluster_separation(self, binary_mask, radius=1):
        coords = np.argwhere(binary_mask)
        if len(coords) == 0:
            return np.zeros_like(binary_mask, dtype=int), 0
        
        clustering = DBSCAN(eps=radius, min_samples=1).fit(coords)
        labels = clustering.labels_

        
        labeled_mask = np.zeros_like(binary_mask, dtype=int)
        for label, (y, x) in zip(labels, coords):
            labeled_mask[y, x] = label + 1  # Start from label 1
        

        return labeled_mask, labels.max() + 1



    def _get_masked(self, is_sci, sigma = 3):
        """
        Generate a masked version of the image (science or reference).
        """

        has_host = self.__pixel_coords_host is not None
        data = self.sci_data.copy() if is_sci else self.ref_data.copy()


        # Step 1: Initial sigma clipping and binary mask
        clipped = sigma_clip(data, sigma=sigma, maxiters=3)

        binary_mask = clipped.mask | self.__fixed_center


        # Step 2: Apply smoothing and connected component labeling

        w1, num_labels = self._spatial_cluster_separation(binary_mask)

        noise = data[~binary_mask]
        noise = noise[~np.isnan(noise)]
        label_map = w1.astype(bool)

        mask_3 = None
        # Step 3: Choose masks based on image type (sci/ref)
        if is_sci:
            # self.sci_noise = get_noise_distribution(data, factor1 = noise_factor1, factor2 = noise_factor2)
            self.sci_noise = noise
            if has_host:
                if num_labels == 1:
                    mask_3 = label_map
                else:
                    self.target_region = self._get_closest_cluster(self.__pixel_coords_target, w1, num_labels)
                    self.host_region = self._get_closest_cluster(self.__pixel_coords_host, w1, num_labels)
                    mask_3 = self.target_region + self.host_region
            else:
            
                self.target_region = self._get_closest_cluster(self.__pixel_coords_target, w1, num_labels)
                
                mask_3 = self.target_region 
                
            self.sci_mask = mask_3 > 0
            self.sci_w = w1.astype(bool) 
            
        else:
            # self.ref_noise = get_noise_distribution(data, factor1 = noise_factor1, factor2 = noise_factor2)
            self.ref_noise = noise
            if has_host:
                if num_labels == 1:
                    mask_3 = label_map
                else:
                    self.host_region = self._get_closest_cluster(self.__pixel_coords_host, w1, num_labels)
                    mask_3 = self.host_region
            else:
                mask_3 = self.__fixed_center
            self.ref_mask = mask_3 > 0

            self.ref_w = w1.astype(bool)


        
    def _fuse_masks(self, mask1, mask2):
        # Ensure masks are boolean arrays
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)

        # Combine the two masks using logical OR
        combined_mask = np.logical_or(mask1, mask2)

        return combined_mask       
    
    def _get_noise_clusters(self, unmasked_regions, w):
        w1 = w.copy()

        # Ensure binary masks
        mask1 = unmasked_regions.astype(bool)
        mask2 = w1.astype(bool)

        # Label connected components in both masks
        labeled_mask2, num_features2 = label(mask2)

        # Create an empty mask for the output
        output_mask = np.zeros_like(mask2, dtype=bool)

        # Loop through each cluster in mask2
        for i in range(1, num_features2 + 1):
            cluster_mask = (labeled_mask2 == i)
            
            # Check if this cluster overlaps with any cluster in mask1
            if not np.any(mask1 & cluster_mask):
                # If no overlap, include it in the output
                output_mask |= cluster_mask

        return output_mask.astype(bool)  # or keep as bool
      

    def _get_masked_img(self, sigma = 3, grow_itr = 2):

        if self.ref_mask is None:
            self._get_masked(is_sci = False, sigma = sigma)
        if self.sci_mask is None:
            self._get_masked(is_sci = True, sigma = sigma)

        if self.sci_mask is not None and self.ref_mask is not None:
            self.mask = self._fuse_masks(self.sci_mask, self.ref_mask)

        sum_mask = self._fuse_masks(self.sci_w, self.ref_w)
        sum_nearby_mask = self._get_noise_clusters(unmasked_regions = self.mask, w = sum_mask)
        grown_nearby_mask = binary_dilation(sum_nearby_mask, iterations=grow_itr) 

        if self._display:
            plt.figure(figsize=(2,2))
            plt.imshow(grown_nearby_mask)
            plt.title('grown_nearby_mask')
            plt.show()

        self._mask_nearby_sources(grown_nearby_mask, is_sci = True)
        self._mask_nearby_sources(grown_nearby_mask, is_sci = False)
        
        
    
        if self._display:
            show_images(
                self.sci_data, self.masked_sci_data,
                titles=[f"original sci image", f"masked sci image (score: %.3f" % self.quality_check.run(self.masked_sci_data.copy()) + ")"],
                global_scale=False
            )
            show_images(
                self.ref_data, self.masked_ref_data,
                titles=[f"original ref image", f"masked ref image (score: %.3f" % self.quality_check.run(self.masked_ref_data.copy()) + ")"],
                global_scale=False
            )

        self._match_scaling()
        



    @property
    def check_host_position(self):
        """
        check whether the pixels of host reach to the edges of the image.
        """

        flag = False
        if self.host_region is None:
            return False
        
        i = 0
        while i < 60:
            j = self.host_region[i]
            if i == 0 or i == 59:
                if np.any(j):
                    flag = True 
                    break
            else:
                if j[0] or j[-1]:
                    flag = True 
                    break
            i += 1
   
        return flag
    
 


    def _simulate_missing_data(self, image, vacancy = 10, factor1 = 2, factor2 = 1):
        '''
        This function is to simulate the missing data in the image after masking all the bright stars
        '''
        noise_values = get_noise_distribution(image, factor1 = factor1, factor2 = factor2)
        return np.random.choice(noise_values, size=vacancy, replace=True)
     
 

    def _match_scaling(self):
        '''
        This function is to match the density of the masked image to the original image
        '''
        if self.masked_sci_data is None or self.masked_ref_data is None:
            raise ValueError('No masked image found')
        else:
            sci_std = np.nanstd(self.masked_sci_data)
            ref_std = np.nanstd(self.masked_ref_data)
            scale_factor = sci_std / ref_std
            if self._display:
                show_images(self.masked_sci_data, self.masked_ref_data, titles=['masked sci image', 'masked ref image'], global_scale=True)
            self.masked_ref_data = (self.masked_ref_data - np.nanmean(self.masked_ref_data)) * scale_factor + np.nanmean(self.masked_sci_data)
            # self.masked_ref_data = (self.masked_ref_data - np.nanmedian(self.masked_ref_data)) * scale_factor + np.nanmedian(self.masked_sci_data)
            if self._display:
                show_images(self.masked_sci_data, self.masked_ref_data, titles=['adjusted masked sci image', 'adjusted masked ref image'], global_scale=True)
    
    
    def _flip_image(self, axis = None):
        '''
        This function is to flip the image randomly
        '''
        if axis is None:
            axis = np.random.randint(2)
        flipped_sci = None
        flipped_ref = None
        if self.masked_sci_data is not None and self.masked_ref_data is not None:
            flipped_sci = np.flip(self.masked_sci_data, axis=axis)
            flipped_ref = np.flip(self.masked_ref_data, axis=axis)
        else:
            print('No masked image found')
        
        if self._display:
            show_images(self.masked_sci_data, flipped_sci, titles=['sci image', 'flipped image with axis: ' + str(axis)])
            show_images(self.masked_ref_data, flipped_ref, titles=['ref image', 'flipped image with axis: ' + str(axis)])

        return flipped_sci, flipped_ref



    def _rotate_image(self, angle = None):
        """
        Rotate a square image by a given angle and fill missing edges with sampled noise.
        
        Parameters:
            image (np.ndarray): 2D square image array.
            angle (float): Rotation angle in degrees (counter-clockwise).
           
        Returns:
            np.ndarray: Rotated image with noise-filled edges.
        """

        def rotate_single_image(image):
            if image.ndim != 2 or image.shape[0] != image.shape[1]:
                print("Input image must be a square 2D array.")
                return None
        
            rotated = rotate(image, angle, reshape=False, order=1, mode='constant', cval=np.nan)
            noise = self._simulate_missing_data(rotated, factor1 = noise_factor1, factor2 = noise_factor2)
            noise_sample = np.random.choice(noise, size=rotated.shape, replace=True)
            filled_rotated = np.where(np.isnan(rotated), noise_sample, rotated)
            return filled_rotated

        if angle is None:
            angle = np.random.randint(360)
        if self.check_host_position and angle%90 != 0:
            angle = 90 * np.random.randint(4)
         
        print('rotation angle: ', angle)
        rotated_sci = rotate_single_image(self.masked_sci_data)
        rotated_ref = rotate_single_image(self.masked_ref_data)

        if self._display:
            show_images(self.masked_sci_data, rotated_sci, titles=['sci image', 'rotated image with angle: ' + str(angle)])
            show_images(self.masked_ref_data, rotated_ref, titles=['ref image', 'rotated image with angle: ' + str(angle)])
        
        return rotated_sci, rotated_ref


    



    def display_results(self):
        
        self._get_masked_img(sigma = 2)
        flipped_sci, flipped_ref = self._flip_image()
        rotated_sci, rotated_ref = self._rotate_image()
      

        
     

