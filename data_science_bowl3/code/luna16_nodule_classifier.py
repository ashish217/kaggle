import os
import sys
import glob

import numpy as np
import pandas as pd

import SimpleITK as sitk
from scipy.ndimage import rotate, imread
from PIL import Image
from scipy.misc import imread
from sklearn.cross_validation import train_test_split

import tensorflow as tf
import tflearn
from tflearn.data_utils import build_hdf5_image_dataset
import h5py

from joblib import Parallel, delayed

from cnn_model import CNNModel
from ctscan import CTScan

"""Most of the code in this file is adapted from:
    https://github.com/swethasubramanian/LungCancerDetection
    All credit goes to the original author swetha subramanian.
"""

def create_data(idx, outDir, X_data,  raw_image_path, width = 50):
    '''
    Generates your test, train, validation images
    outDir = a string representing destination
    width (int) specify image size
    '''
    outfile = outDir  +  str(idx)+ '.jpg'
    if not os.path.isfile(outfile):
        scan = CTScan(np.asarray(X_data.loc[idx])[0], np.asarray(X_data.loc[idx])[1:], raw_image_path)
        scan.save_image(outfile, width)


def do_test_train_split(filename):
    """
    Does a test train split if not previously done
    """
    candidates = pd.read_csv(filename)

    positives = candidates[candidates['class']==1].index  
    negatives = candidates[candidates['class']==0].index

    ## Under Sample Negative Indexes
    np.random.seed(974)
    negIndexes = np.random.choice(negatives, len(positives)*5, replace=False)

    candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]

    X = candidatesDf.iloc[:,:-1]
    y = candidatesDf.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 42)

    X_train.to_pickle('traindata')
    y_train.to_pickle('trainlabels')
    X_test.to_pickle('testdata')
    y_test.to_pickle('testlabels')
    X_val.to_pickle('valdata')
    y_val.to_pickle('vallabels')

def augment_positive_cases(idx, outDir):
    inp = imread(outDir + str(idx) + '.jpg')
    # Rotate by 90
    inp90 = rotate(inp, 90, reshape = False)
    Image.fromarray(inp90).convert('L').save(outDir + str(idx+1000000) + '.jpg')

    inp180 = rotate(inp, 180, reshape = False)
    Image.fromarray(inp180).convert('L').save(outDir + str(idx+2000000) + '.jpg')

def create_hdfs_files(luna_output):
    for mode in ['train', 'test', 'val']:
        print ("current mode is", mode)
        X = pd.read_pickle(mode + 'data')
        y = pd.read_pickle(mode + 'labels')

        output_dir = luna_output + mode + '/'

        filenames = X.index.to_series().apply(lambda x: output_dir + str(x) + '.jpg')
        filenames = filenames.values.astype(str)
        labels = y.values.astype(int)

        data = np.zeros(filenames.size, dtype=[('var1', 'S100'), ('var2', int)])
        data['var1'] = filenames
        data['var2'] = labels

        dataset_file = luna_output + mode + 'datalabels.txt'
        np.savetxt(dataset_file, data, fmt="%s %d")

        output = luna_output + mode + 'dataset.h5'

        build_hdf5_image_dataset(dataset_file, 
                                 image_shape = (50, 50, 1), 
                                 mode ='file', 
                                 output_path = output, 
                                 categorical_labels = True, 
                                 normalize = True,
                                 grayscale = True)

        # Load HDF5 dataset
        h5f = h5py.File(output, 'r')
        X_images = h5f['X']
        Y_labels = h5f['Y'][:]

        print (X_images.shape)
        X_images = X_images[:,:,:].reshape([-1,50,50,1])
        print (X_images.shape)
        h5f.close()

        h5f = h5py.File(luna_output + mode + '.h5', 'w')
        h5f.create_dataset('X', data=X_images)
        h5f.create_dataset('Y', data=Y_labels)
        h5f.close()
    
def train_cnn_model(luna_output, OUTPUT_MODEL_FILE):
    # Load HDF5 dataset
    h5f = h5py.File(luna_output + 'train.h5', 'r')
    X_train_images = h5f['X']
    Y_train_labels = h5f['Y']

    h5f2 = h5py.File(luna_output + 'val.h5', 'r')
    X_val_images = h5f2['X']
    Y_val_labels = h5f2['Y']

    tf.reset_default_graph()
    convnet  = CNNModel()

    network = convnet.define_network(X_train_images)
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='nodule3-classifier.tfl.ckpt')

    model.fit(X_train_images, Y_train_labels, n_epoch = 100, shuffle=True,
              validation_set = (X_val_images, Y_val_labels), show_metric = True,
              batch_size = 96, snapshot_epoch = False, run_id = 'nodule3-classifier')

    model.save(OUTPUT_MODEL_FILE)
    
    h5f.close()
    h5f2.close()
    
if __name__ == '__main__':
    '''
    Arg 1: Input top directory for LUNA16 data
    Arg 2: Output directory where intermedia data while be stored 
    Example args: /notebooks/DSB3/data/LUNA/ /notebooks/ashish/luna16_patch_2/
    '''
    if len(sys.argv) < 3:
        print( 'Usage: ', sys.argv[0], ' input_luna16_dir output_dir' )
        sys.exit(0)
        
    luna_path = sys.argv[1]
    raw_image_path = luna_path + '/*/'
    luna_output = sys.argv[2]

    annotations = pd.read_csv(luna_path + "CSVFILES/" + "annotations.csv")
    candidates = pd.read_csv(luna_path + "CSVFILES/" + "candidates.csv")

    #####################################################
    ## extract small image chip around known nodules and store it in binary file
    candidates_file = luna_path + "CSVFILES/" + "candidates.csv"
    do_test_train_split(candidates_file)

    for mode in ['test', 'train', 'val']:
        output_dir = luna_output + mode + '/'
        X_data = pd.read_pickle(mode + 'data')
        Parallel(n_jobs = 10)(delayed(create_data)(idx, output_dir, X_data, raw_image_path) for idx in X_data.index)
        #for idx in X_data.index:
        #    create_data(idx, output_dir, X_data, raw_image_path)

    print ("Finished cropping images around nodules")

    #####################################################
    ## augment training data
    X_train = pd.read_pickle('traindata')
    y_train = pd.read_pickle('trainlabels')
    augIndexes = X_train[y_train == 1].index

    mode = 'train'
    output_dir = luna_output + mode + '/'
    Parallel(n_jobs = 10)(delayed(augment_positive_cases)(idx, output_dir) for idx in augIndexes)
    print ("Finished data augmentation")

    #####################################################
    ## create HDFS file for train, test and validation prior to feed into NN

    create_hdfs_files(luna_output)
    print ("Finished generating test, train, val hdfs files. Ready for training!")
    #####################################################
    ## Train a NN model and save it to a file

    OUTPUT_MODEL_FILE = 'nodule3-classifier.tfl'
    train_cnn_model(luna_output, OUTPUT_MODEL_FILE)

    print("Network trained and saved as ", OUTPUT_MODEL_FILE)

