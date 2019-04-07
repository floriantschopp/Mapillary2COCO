"""
    This file shows how to load and use the dataset
"""

from __future__ import print_function

import json
import os

import numpy as np
# matplotlib.use('Agg')
import scipy.misc
from PIL import Image

from multiprocessing import Pool, freeze_support
import itertools

def split_to_coco_creator(input_instance_array, labels):

    labelid_matrix_name = []

    label_image_info = np.array(input_instance_array / 256, dtype=np.uint8)

    instance_image_info = np.array(input_instance_array % 256, dtype=np.uint8)


    unique_label_info = np.unique(label_image_info)
    unique_instance_info = np.unique(instance_image_info)

    for label_id, label in enumerate(labels):

        if (label_id in (unique_label_info)) and (label["instances"] == True):

            each_label_array = np.zeros((input_instance_array.shape[0], input_instance_array.shape[1]),
                                        dtype=np.uint8)

            each_label_array[label_image_info == label_id] = 255

            for instance_id in range(256):
                if (instance_id in unique_instance_info):
                    each_instance_array = np.zeros(
                        (input_instance_array.shape[
                         0], input_instance_array.shape[1]),
                        dtype=np.uint8)

                    each_instance_array[
                        instance_image_info == instance_id] = 255

                    final_instance_array = np.bitwise_and(
                        each_instance_array, each_label_array)

                    if np.unique(final_instance_array).size == 2:
                        labelid_matrix_name.append(
                            {"label_id": label_id, "instance_id": instance_id,
                             "label_name": label["readable"],
                             "image": final_instance_array})

    # each_id_array [(input_instance_array % 256) == instance_id] = 1
    # labelid_matrix_name.append ( (label_id , instance_id, label [ "readable"
    # ] , each_id_array) )

    return labelid_matrix_name

def process_file(file_name, dir_name, labels):
    print("Task {}-{}".format(file_name,dir_name))
    file_name = file_name[:-4]
    instance_path = "{}/instances/{}.png".format(dir_name,file_name)
    instance_image = Image.open(instance_path)
    instance_array = np.array(instance_image, dtype=np.uint16)
    image_label_instance_infomatrix = split_to_coco_creator(
        instance_array, labels)

    for item in image_label_instance_infomatrix:
        path = "{}_{}_{}.png".format(
            file_name, item["label_name"].replace(" ", "_"), item["instance_id"])
        scipy.misc.imsave("{}/annotations/{}".format(dir_name,path), item["image"])

def process_file_helper(args):
    return process_file(*args)

def split_dir(dir_name):
    pool = Pool()
    print ("Spliting {}".format(dir_name))
    if not os.path.exists("{}/annotations".format(dir_name)):
        os.makedirs("{}/annotations".format(dir_name))
    dir_path = "{}/instances".format(dir_name)
    files = os.listdir(dir_path)
    # read in config file
    with open('config.json') as config_file:
        config = json.load(config_file)

    labels = config['labels']
    
    pool.map(process_file_helper, itertools.izip(files, 
        itertools.repeat(dir_name), itertools.repeat(labels)))
        
if __name__ == '__main__':
    freeze_support()
    split_dir("training")
