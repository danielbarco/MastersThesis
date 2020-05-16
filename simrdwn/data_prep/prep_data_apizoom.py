#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 06:14:20 2019

@author: avanetten

Parse COWC dataset for SIMRDWN training

Data located at:
    https://gdo152.llnl.gov/cowc/
    cd /raid/data/
    wget -r -np  ftp://gdo152.ucllnl.org/cowc/datasets/ground_truth_sets

"""


import os
import glob
import sys
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
sys.path.insert(0, os.path.dirname(os.path.abspath('.'))) # to make imports relative to project root work
import shutil
import importlib
import pickle

import parse_apizoom
import yolt_data_prep_funcs

path_simrdwn_utils = os.getcwd()
path_simrdwn_utils = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path_simrdwn_utils, '..', 'core'))
import preprocess_tfrecords


###############################################################################
# path variables (may need to be edited! )

# gpu07
apizoom_data_dir = 'simrdwn/data/apizoom_ground_truth_SCLD/'
label_map_file = 'class_labels_varroa.pbtxt'
verbose = True

# at /simrdwn
simrdwn_data_dir = '/simrdwn/data/train_data'
label_path_root = '/simrdwn/data/train_data/'
annotations_path_root = apizoom_data_dir + 'Annotations/'
train_images_path = apizoom_data_dir + 'train/'
#folder_name = 'apizoom_SCLD_1500'
# file_path = '../ApiZoom_SIMRDWN_dataIN/' + folder_name + '/'
# label_path_root = file_path + 'Annotations/'
# file_path_images = file_path + 'images/'
train_out_dir = '/simrdwn/data/train_data/apizoom_1500_overlay'
test_out_dir = '/simrdwn/data/test_images/apizoom_1500_overlay'

# at /cosmiqyx                          
# simrdwn_data_dir = '/cosmiq/src/simrdwn3/data/train_data'
# label_path_root = '/cosmiq/src/simrdwn3/data/train_data'
# train_out_dir = '/cosmiq/src/simrdwn3/data/train_data/cowc'
# test_out_dir = '/cosmiq/src/simrdwn3/data/test_images/cowc'
# at /local_data
# simrdwn_data_dir = '/local_data/simrdwn3/data/train_data'
# label_path_root = '/local_data/simrdwn3/data/train_data'
# test_out_dir = '/local_data/simrdwn3/data/test_images/cowc'

label_map_path = os.path.join(label_path_root, label_map_file)
print ("label_map_path:", label_map_path)


##############################
# list of train and test directories
# for now skip Columbus and Vahingen since they are grayscale
# os.path.join(args.cowc_data_dir, 'datasets/ground_truth_sets/')
ground_truth_dir = apizoom_data_dir
train_dirs = ['train']
test_dirs = ['test']
test_suffix = 'test'
annotation_suffix = '_Annotated.png'
##############################

##############################
# infer training output paths
labels_dir = os.path.join(train_out_dir, 'labels/')
images_dir = os.path.join(train_out_dir, 'images/')
im_list_name = os.path.join(train_out_dir, 'apizoom_1500_overlay_train_list.txt')
tfrecord_train = os.path.join(train_out_dir, 'apizoom_1500_overlay_train.tfrecord')
sample_label_vis_dir = os.path.join(train_out_dir, 'sample_label_vis/')
# im_locs_for_list = output_loc + train_name + '/' + 'training_data/images/'
# train_images_list_file_loc = yolt_dir + 'data/'
# create output dirs
for d in [train_out_dir, test_out_dir, labels_dir, images_dir]:
    if not os.path.exists(d):
        print("make dir:", d)
        os.makedirs(d)
##############################

##############################
# set yolt training box size
car_size = 3      # meters
GSD = 0.15        # meters
##yolt_box_size = np.rint(car_size/GSD)  # size in pixels
yolt_box_size = 32
print("yolt_box_size (pixels):", yolt_box_size)
##############################

##############################
# slicing variables

zero_frac_thresh = 0.2
sliceHeight, sliceWidth = 1500, 1500  # for for 82m windows
slice_overlap = 32 / sliceHeight
##############################

#############################
#set yolt category params from pbtxt
label_map_dict = preprocess_tfrecords.load_pbtxt(label_map_path, verbose=False)
print("label_map_dict:", label_map_dict)
# get ordered keys
key_list = sorted(label_map_dict.keys())
category_num = len(key_list)
# category list for yolt
cat_list = [label_map_dict[k] for k in key_list]
print("cat list:", cat_list)
yolt_cat_str = ','.join(cat_list)
print("yolt cat str:", yolt_cat_str)
# create yolt_category dictionary (should start at 0, not 1!)
yolt_cat_dict = {x: i for i, x in enumerate(cat_list)}
print("yolt_cat_dict:", yolt_cat_dict)
# conversion between yolt and pbtxt numbers (just increase number by 1)
convert_dict = {x: x+1 for x in range(100)}
print("convert_dict:", convert_dict)
##############################


##############################
# Get labels from XML
##############################
def xml_to_df(path):
    xml_list = []
    #for  in glob.glob(path + '/*.xml'):
    for xml_file in listdir(path):
        filename, file_extension = os.path.splitext(xml_file)
        #print(filename)
        #print(xml_file)
        if isfile(join(path, xml_file)) and file_extension == ".xml" and not filename.startswith("._"):
            tree = ET.parse(join(path, xml_file))
            #print(join(path, xml_file))
            root = tree.getroot()
            for member in root.findall('object'):
                value = (filename,
                         #int(root.find('size')[0].text),
                         #int(root.find('size')[1].text),

                         member[0].text,
                         int(member[2][0].text),
                         int(member[2][1].text),
                         int(member[2][2].text),
                         int(member[2][3].text),
                         )
                xml_list.append(value)
    column_name = ['filename','class', 'x1', 'y1', 'x2', 'y2'] # 'width', 'height',
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
##############################


##############################
# Convert to YOLO
##############################
# The following code is the modified version of codes available here: 
# https://blog.goodaudience.com/part-1-preparing-data-before-training-yolo-v2-and-v3-deepfashion-dataset-3122cd7dd884

def convert_labels(path, x1, y1, x2, y2):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
    """
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    size = get_img_shape(path)
    #print(size)
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def get_img_shape(path):
    #path = 'images/'+path
    img = cv2.imread(path)
    try:
        return img.shape
    except AttributeError:
        print('error! ', path)
        return (None, None, None)
    
##############################



##############################
# Slice large images into smaller chunks
##############################
print("im_list_name:", im_list_name)
if os.path.exists(im_list_name):
    print('RUN SLICE FALSE')
    run_slice = False
else:
    run_slice = True
    print('RUN SLICE TRUE')
    
df = xml_to_df(annotations_path_root)
print(len(df['filename'].unique()))

df['org_img'] = df['filename'].str.replace(r"_32px.*","")
df['cut_nr'] = pd.to_numeric(df['filename'].str.replace(r".*(?<=_32px_)", '').str.strip())
df['path_img_cut'] =  images_dir + df['filename'] + '.jpg'
df['path_img_org'] =  images_dir + df['org_img'] + '.jpg'

# df['x'], df['y'], df['width'], df['height'] = \
# zip(*df.apply(lambda row: convert_labels(row['path_img_cut'], row['x1'], row['y1'], row['x2'], row['y2']), axis = 1))

test = df[df['filename'].str.contains('test', regex=True)==True]
train = df[df['filename'].str.contains('test', regex=True)==False]
data_sets = {'test': test, 'train': train}

dict_overlay = {}

for sets, data in data_sets.items():
    for filename in train.filename.unique():
        print(filename)
        dtot = train_images_path
        cut_file_tot = os.path.join(dtot, filename + '.jpg')
        outroot =  sets + '_' + filename.split('.')[0]
        box_coords = list(df[df['filename'] == filename].apply(lambda row: [row.x1, row.x2, row.y1, row.y2], axis = 1))
        print(' dtot: ', dtot)
        # print(' cut_file_tot: ', cut_file_tot)
        # print(' outroot: ', outroot)
        # print(' labels_dir: ', labels_dir)
        # print(' images_dir: ', images_dir)
       
        if run_slice:
            parse_apizoom.slice_im_apizoom(
                cut_file_tot, 
                outroot, images_dir, labels_dir, yolt_cat_dict, cat_list[0],
                box_coords, dict_overlay,
                sliceHeight=sliceHeight, sliceWidth=sliceWidth,
                zero_frac_thresh=zero_frac_thresh, overlap=slice_overlap,
                pad=0, verbose=verbose) 
            
if len(dict_overlay) > 0:
    outname_pkl = os.path.join(train_out_dir, 'dict_overlay.pkl')
    pickle.dump(dict_overlay, open(outname_pkl, 'wb'), protocol=2)
            
##############################

##############################
# Get list for simrdwn/data/, copy to data dir
##############################
train_ims = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
f = open(im_list_name, 'w')
for item in train_ims:
    f.write("%s\n" % item)
f.close()
# copy to data dir
print("Copying", im_list_name, "to:", simrdwn_data_dir)
shutil.copy(im_list_name, simrdwn_data_dir)
##############################

##############################
# Ensure labels were created correctly by plotting a few
##############################
max_plots = 50
thickness = 2
yolt_data_prep_funcs.plot_training_bboxes(
    labels_dir, images_dir, ignore_augment=False,
    sample_label_vis_dir=sample_label_vis_dir,
    max_plots=max_plots, thickness=thickness, ext='.jpg')


##############################
# Make a .tfrecords file
##############################
importlib.reload(preprocess_tfrecords)
preprocess_tfrecords.yolt_imlist_to_tf(im_list_name, label_map_dict,
                                       TF_RecordPath=tfrecord_train,
                                       TF_PathVal='', val_frac=0.0,
                                       convert_dict=convert_dict, verbose=True)
# copy train file to data dir
print("Copying", tfrecord_train, "to:", simrdwn_data_dir)
shutil.copy(tfrecord_train, simrdwn_data_dir)



##############################
# Copy test images to test dir
print("Copying test images to:", test_out_dir)
for td in test_dirs:
    td_tot_in = os.path.join(ground_truth_dir, td)
    td_tot_out = os.path.join(test_out_dir, td)
    if not os.path.exists(td_tot_out):
        os.makedirs(td_tot_out)
    # copy non-label files
    for f in os.listdir(td_tot_in):
        if f.endswith('.jpg') and not f.endswith(('_Cars.png', '_Negatives.png', '.xcf')):
            shutil.copy2(os.path.join(td_tot_in, f), td_tot_out)
    # copy everything?
    #os.system('cp -r ' + td_tot + ' ' + test_out_dir)
    ##shutil.copytree(td_tot, test_out_dir)
##############################
