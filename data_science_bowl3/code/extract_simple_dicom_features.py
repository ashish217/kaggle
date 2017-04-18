from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import os
import dicom


def verify_data(df, INPUT_DATA_DIR, INPUT_DATA_NPY_DIR):
    for id in df.id:
        dicom_folder_path = os.path.join(INPUT_DATA_DIR, id)
        if not os.path.isdir(dicom_folder_path):
            print ("Didn't find dicom folder: ", dicom_folder_path)

        binary_file_path = os.path.join(INPUT_DATA_NPY_DIR, id) + '.npy'
        if not os.path.isfile(binary_file_path):
            print ("Didn't find file: ", binary_file_path)
            
def extract_num_slices(INPUT_DATA_NPY_DIR, row):
    file_path = os.path.join(INPUT_DATA_NPY_DIR, row['id']) + '.npy'
    
    return np.load(file_path).shape[0]  

def extract_metadata(INPUT_DATA_DIR, row):
    folder_path = os.path.join(INPUT_DATA_DIR, row['id'])
    
    slices = [dicom.read_file(folder_path + '/' + s) for s in os.listdir(folder_path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    return pd.Series([len(slices), slice_thickness, float(slices[0].PixelSpacing[0]), slices[0].PixelRepresentation])

def calc_area(in_dcm, min_val, max_val):
    pix_area = np.prod(in_dcm.PixelSpacing)
    return pix_area*np.sum((in_dcm.pixel_array>=min_val) & (in_dcm.pixel_array<=max_val))

def extract_blood_area(INPUT_DATA_DIR, row):
    folder_path = os.path.join(INPUT_DATA_DIR, row['id'])

    slices = [dicom.read_file(folder_path + '/' + s) for s in os.listdir(folder_path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    return calc_area(slices[0], 30, 45)


if __name__ == '__main__':
    '''
    Arg 1: Input CSV file with stage1 labels (training data)
    Arg 2: Input CSV file with stage1 sample submission (test data)
    Arg 3: Input directory with raw DICOM folders (top level directory)
    Arg 4: Input directory with NPY files generated for each patient using dcm_to_npy.py script
    Arg 5: Output CSV file will be generated with simple features
    Example args: /notebooks/DSB3/data/stage1_labels.csv /notebooks/ashish/stage1_sample_submission.csv /notebooks/DSB3/data/stage1/stage1/ /notebooks/DSB3/data/stage1_npy/ /notebooks/ashish/simple_features.csv
    '''
    if len(sys.argv) < 6:
        print( 'Usage: ', sys.argv[0], 'stage1_labels.csv stage1_sample_submission.csv input_dir_raw_dicom input_dir_npy output_feature_file' )
        sys.exit(0)
        
    STAGE1_LABELS_FILE = sys.argv[1]
    STAGE1_SUBM_FILE = sys.argv[2]
    INPUT_DATA_DIR = sys.argv[3]
    INPUT_DATA_NPY_DIR = sys.argv[4]
    OUTPUT_FEAT_FILE = sys.argv[5]

    # combine feature for all test and train data
    if not os.path.isfile(OUTPUT_FEAT_FILE):
        df_train = pd.read_csv(STAGE1_LABELS_FILE)
        df_test = pd.read_csv(STAGE1_SUBM_FILE)
        df = pd.concat([df_train,df_test])
        print("Train: %d, Test: %d, All: %d" % (len(df_train), len(df_test), len(df)))
        print("Created new file: %s" % OUTPUT_FEAT_FILE)
        
        ## remove 'cancer' column since we are only collection features here
        df.drop('cancer', axis=1, inplace=True)
        df.to_csv(OUTPUT_FEAT_FILE, index=False)

    ## read the most updated file
    df = pd.read_csv(OUTPUT_FEAT_FILE)
            
    ## Just making sure we have all the necessary data to load from both directories
    verify_data(df, INPUT_DATA_DIR, INPUT_DATA_NPY_DIR)
    print ("Data verification completed!")
        
    ## extract number of slices
    if 'slices' not in df.columns:
        df['slices'] = df.apply(lambda row: extract_num_slices(INPUT_DATA_NPY_DIR, row), axis=1)   
        
    ## image resolution related features
    if 'slice_thick' not in df.columns:
        df[['slices','slice_thick', 'pixel_spacing', 'pixel_repre']] = df.apply(lambda row: extract_metadata(INPUT_DATA_DIR, row), axis=1)
        df['height'] = df['slice_thick']*df['slices']

    ## blood area calculated based on HU range (30 - 45)
    if 'blood_area' not in df.columns:
        df['blood_area'] = df.apply(lambda row: extract_blood_area(INPUT_DATA_DIR, row), axis=1)
        
    # write out features to the output file
    df.to_csv(OUTPUT_FEAT_FILE, index=False)
    print ("All extracted features stored in file", OUTPUT_FEAT_FILE)
        