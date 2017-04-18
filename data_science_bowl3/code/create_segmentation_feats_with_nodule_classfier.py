import glob, random
import os
import sys
import numpy as np
import pandas as pd

from skimage.morphology import ball
from skimage import measure, feature
from skimage.morphology import binary_dilation, binary_opening, binary_closing
from skimage.filters import roberts, sobel
from skimage.measure import label,regionprops, perimeter

from sklearn import cross_validation
import xgboost as xgb
import tensorflow as tf
import tflearn
from joblib import Parallel, delayed

from cnn_model import CNNModel


###########################################
def get_subimage(image, center, width):
    """
    Returns cropped image of requested dimensiona
    """
    z, y, x = center
    subImage = image[int(z), int(y-width/2):int(y+width/2), int(x-width/2):int(x+width/2)]
    return subImage.copy()

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def segment_nodules(segmented_ct_scan):
    """Simple threshold based segmentation"""
    segmented_ct_scan[segmented_ct_scan < -30] = 0
    segmented_ct_scan[segmented_ct_scan > 400] = 0
    
    selem = ball(2)
    binary = binary_closing(segmented_ct_scan, selem)
    
    return binary

def can_crop_image(image, center, width):
    z, y, x = center
    if int(z) >= image.shape[0] or int(y-width/2) < 0 or int(y+width/2) >= image.shape[1] or int(x-width/2) < 0 or int(x+width/2) >= image.shape[2]:
        return False
    else:
        return True

def contains_nodule(region, lungs):
    """check if it classified as nodule based on model built using luna16 data"""
    center = region.coords.mean(axis=0)
    if can_crop_image(lungs, center, IMG_SIZE):
        img_chip = get_subimage(lungs, center, IMG_SIZE)
        img_chip_norm = normalizePlanes(img_chip)
    else:
        return False, []
    
    if img_chip_norm.shape[0] == IMG_SIZE and img_chip_norm.shape[1] == IMG_SIZE:
        prediction = model.predict(img_chip_norm.reshape((1,img_chip_norm.shape[0], img_chip_norm.shape[1], 1)))
        fc_feats = fc_model.predict(img_chip_norm.reshape((1,img_chip_norm.shape[0], img_chip_norm.shape[1], 1)))
        return prediction[0][1] > 0.92, fc_feats[0]
    else:
        return False, []
###########################################
def segmented_nodules_feats(segmented_ct_scan):
    """This function segments nodules and extract nodule features"""
    binary = segment_nodules(segmented_ct_scan.copy())
    label_scan = label(binary)
    regions = regionprops(label_scan, segmented_ct_scan)

    areas = [r.area for r in regions]
    areas.sort()
    ############################
    
    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512 
    eqDiameters = []
    final_regions = []
    fc_feat_list = []
    
    ## examine each 3d regions
    for r in regions:
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000

        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)

            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
            
        has_nodule, fc_feats = contains_nodule(r, segmented_ct_scan)
        if (min_z == max_z or min_y == max_y or min_x == max_x 
            or r.area > areas[-3] 
            or (max_z - min_z) < 1 
            or (max_z - min_z) > 30 
            or not has_nodule):
            for c in r.coords:
                #segmented_ct_scan[c[0], c[1], c[2]] = 0
                binary[c[0], c[1], c[2]] = 0
        else:
            final_regions.append(r)
            fc_feat_list.append(fc_feats)
            
        
    if len(final_regions):
        numNodes = len(final_regions) + 1
        
        ## area based feat
        final_areas = [r.area for r in final_regions] #+ [0]
        minArea = min(final_areas)
        maxArea = max(final_areas)
        totalArea = sum(final_areas)
        avgArea = totalArea / numNodes
        stdAreas = np.std(final_areas)

        # equivalent_diameter The diameter of a circle with the same area as the region.
        eqDiameters = [r.equivalent_diameter for r in final_regions] #+ [0]
        avgEquivlentDiameter = sum(eqDiameters) / numNodes
        stdEquivlentDiameter = np.std(eqDiameters)
        minEquivlentDiameter = min(eqDiameters)
        maxEquivlentDiameter = max(eqDiameters)
        
        meanIntensities = [r.mean_intensity for r in final_regions]
        avgMeanIntensities = sum(meanIntensities) / numNodes
        stdMeanIntensities = np.std(eqDiameters)
        minMeanIntensities = min(eqDiameters)
        maxMeanIntensities = max(eqDiameters)
        
        minIntensities = [r.min_intensity for r in final_regions]
        avgMinIntensities = sum(minIntensities) / numNodes
        
        maxIntensities = [r.max_intensity for r in final_regions]
        avgMaxIntensities = sum(maxIntensities) / numNodes
        
        fc_feat = np.mean(fc_feat_list, axis=0).tolist()

        return binary, segmented_ct_scan, np.array([totalArea, avgArea, maxArea, avgEquivlentDiameter,\
                         stdEquivlentDiameter, numNodes, stdAreas, minArea, minEquivlentDiameter, maxEquivlentDiameter,
                         avgMeanIntensities, stdMeanIntensities, minMeanIntensities, maxMeanIntensities,    
                         avgMinIntensities, avgMaxIntensities] + fc_feat)
    else:
        return binary, segmented_ct_scan, np.zeros(16 + 512)
    
###########################################
def extract_and_save_features(input_file, OUTPUT_DATA_DIR):
    out_file_path = os.path.join(OUTPUT_DATA_DIR, os.path.basename(input_file))
    if os.path.isfile(out_file_path):
        print(out_file_path + " already created.")
    else:
        print ("Creating file: %s" % out_file_path)
        orig_data = np.load(input_file)
        tmp, tmp1, feats = segmented_nodules_feats(orig_data)
        np.save(out_file_path, feats)  
        
###########################################
def generate_feature_csv(FEATURE_DIR):
    feat_list = []
    pid_list = []
    for input_file in glob.glob(FEATURE_DIR + '/*.npy'):
        feat_list.append(np.load(input_file))
        pid_list.append(os.path.basename(input_file)[:-4])

    ## add column names for all of those features
    columns=['totArea', 'avgArea', 'maxArea', 'avgEquDiam', 'stdEquDiam', 
             'numNodes', 'stdArea', 'minArea', 'minEquDiam', 'maxEquDiam',
             'avgMeanInt', 'stdMeanInt', 'minMeanInt', 'maxMeanInt', 'avgMinInt', 'avgMaxInt']
    columns += ['fc_%d' % i for i in range(len(feat_list[0])-len(columns))]
    df = pd.DataFrame(feat_list, columns=columns)
    df['id'] = pid_list

    return df
        
###########################################

if __name__ == '__main__':
    '''
    Arg 1: Input directory with sugmente lung NPY files generated from lung_segmentation_npy.py script
    Arg 2: Input CNN model trained using luna16_nodule_classifier.py
    Arg 3: Output directory where segmented nodule feature will be stored in NPY format (intermedia storage)
    Arg 4: Output CSV file containing features for each patient
    Example args: /notebooks/DSB3/data/stage1_lungs_npy/ /notebooks/ashish/Notebooks/LUNA16_classifier/nodule3-classifier.tfl /notebooks/ashish/seg_feats_4/ /notebooks/ashish/features_csv/luna16_classifier_feats_fc512.csv
    '''
    if len(sys.argv) < 5:
        print( 'Usage: ', sys.argv[0], 'input_file_dir input_cnn_model_file output_feat_dir output_feat_csv_file' )
        sys.exit(0)
        
    INPUT_DATA_DIR = sys.argv[1]
    model_file = sys.argv[2]
    OUTPUT_DATA_DIR = sys.argv[3]
    OUTPUT_FEATURE_FILE = sys.argv[4]

    ###########################################
    ## load CNN model trained on LUNA16 data
    tf.reset_default_graph()
    IMG_SIZE = 50
    input_shape = np.zeros((1, IMG_SIZE, IMG_SIZE, 1))

    convnet = CNNModel()
    fc_layer, network = convnet.define_network(input_shape, 'fc_feat')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load(model_file)

    fc_model = tflearn.DNN(fc_layer, session = model.session)
    print ("Finished loading CNN model from", model_file)

    ###########################################
    #Parallel(n_jobs = 5)(delayed(extract_and_save_features)(input_file) for input_file in glob.glob(INPUT_DATA_DIR + '*.npy'))
    file_list = glob.glob(INPUT_DATA_DIR + '/*.npy')
    random.shuffle(file_list)
    for input_file in file_list:
        extract_and_save_features(input_file, OUTPUT_DATA_DIR)
    print ("Done generating all the features. Now generating feature csv file.")
    
    df = generate_feature_csv(OUTPUT_DATA_DIR)
    df.to_csv(OUTPUT_FEATURE_FILE, index=False)
    print ("This script completed!")