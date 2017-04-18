'''
This script reads 3D CT scan data stored in numpy array and attempts to segment
out lung per 2D slice. The script assumes input pixel data are in HU.
'''

from __future__ import print_function
import sys
import glob
import os
import numpy as np
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from joblib import Parallel, delayed


def get_segmented_lungs(im, plot=False):
    ## Original author of this function ArnavJain
    ## https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/run/973430
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    return im

def segment_lung_from_ct_scan(ct_scan):
    #assumes the input slices are already sorted based on the verticle position
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

def process_single_file(input_file, OUTPUT_DATA_DIR):
    ## perform lung segmentation and save off numpy binary
    out_file_path = os.path.join(OUTPUT_DATA_DIR, os.path.basename(input_file))
    if os.path.isfile(out_file_path):
        print(out_file_path + " already created.")
    else:
        print ("Creating file: %s" % out_file_path)
        orig_data = np.load(input_file)
        lugs = segment_lung_from_ct_scan(orig_data)
        np.save(out_file_path, lugs)

if __name__ == '__main__':
    '''
    Arg 1: Input directory with NPY files generated from dcm_to_npy.py script
    Arg 2: Output directory where segmented lungs will be stored in NPY format
    Example args: /notebooks/DSB3/data/stage1_npy/ /notebooks/DSB3/data/stage1_lungs_npy/
    '''
    if len(sys.argv) < 3:
        print( 'Usage: ', sys.argv[0], 'input_file_dir output_file_dir' )
        sys.exit(0)

    INPUT_DATA_DIR = sys.argv[1]
    OUTPUT_DATA_DIR = sys.argv[2]
    
    Parallel(n_jobs = 10)(delayed(process_single_file)
                          (input_file, OUTPUT_DATA_DIR) for input_file in glob.glob(INPUT_DATA_DIR + '/*.npy'))
    
    print ("Script completed!")
