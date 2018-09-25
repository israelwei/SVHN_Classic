"""
This file is used for converting the data in the digitStruct.mat for the training set,
which is the data from the "train" tar. Extracting the bounding boxes and labels 
for all images in the training set.
"""

import os
import h5py
import numpy as np
import random
from PIL import Image
import cv2
import glob
from skimage.measure import label
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import scipy

DATA_DIR = './data'
DIGIT_PATCHES_DIR = './classes/'
numbers_count = 0

CROP_SIZE = 16
PATCH_NAME = "Patches"

NUM_LABELS = 10
dict_seen_labels = {}


def clean_small_labeled_areas(binary_img, n_biggest=1):
    """
    Remove all connected components that don't belong to the largest connected components
    :param binary_img: binary image from which we want to filter some of the connected components
    :param n_biggest: the number of biggest connected component to return if wanted
    :return: binary image with the n-largest connected components, or a cleaned version (removing small components)
    """

    labeled_components, num_connected_components = \
        label(binary_img, return_num=True)
    id_sizes = np.array(ndimage.sum(binary_img, labeled_components, range(num_connected_components + 1)))
    sorted_comps = np.sort(id_sizes)[::-1]
    if n_biggest:
        n_biggest_comp = sorted_comps[n_biggest - 1]

        area_mask = (id_sizes < n_biggest_comp)
    else:
        area_mask = (id_sizes < 10 )

    binary_img[area_mask[labeled_components]] = 0

    return binary_img



def skeleton_digit(cur_img_gray):
    """
    A possible method for getting the skeleton of the digit. Not used here because I got relatively good results without
    it.
    :param cur_img_gray: original image to get skeleton from
    :return: skeletonization of the input image
    """

    th = cv2.adaptiveThreshold(cur_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

    th = th / 255.0
    th = clean_small_labeled_areas(th, 0)
    # clean also the inverse
    th = 1 - th
    th = clean_small_labeled_areas(th, 0)
    th = 1 - th
    grad = scipy.ndimage.gaussian_gradient_magnitude(th, sigma=0.5) #sometimes I used this gradient image but it wasn't
                                                                    #always useful so I didn't use it in the end
    skeleton = skeletonize(th)
    return skeleton




def append_to_h5_file(table_name, cur_label, cur_patch):
    """
    This function appends to h5 file the new patch and the additional properties, in new group.
    Each digit will be saved in a separate h5 file for easy control and loading later.

    :param table_name: The full path for the h5 file output, with the filename being "class_<number>" where the "number"
                        is an int between 0-9.
    :param cur_label: The current label of the patches to be saved. In each call the same label is relevant for all patches,
                    it's used here for verifying if it's the first time the digit was recognized for creating
                     a new dataset in h5py.
    """


    # Creating dataset for current digit if digit's label was seen for the first time
    if cur_label not in dict_seen_labels:
        if cur_label not in dict_seen_labels:
            dict_seen_labels[cur_label] = 1
        hf = h5py.File(table_name + '.h5', 'w')  # write mode
        hf.create_dataset(PATCH_NAME, data=cur_patch, maxshape=(None, CROP_SIZE), chunks=True)
        hf.close()

    else:
        hf = h5py.File(table_name + '.h5', 'a')  # append mode
        hf[PATCH_NAME].resize((hf[PATCH_NAME].shape[0] + cur_patch.shape[0]), axis=0)
        hf[PATCH_NAME][-cur_patch.shape[0]:] = cur_patch
        hf.close()




def step_process(image, bbox_left, bbox_top, bbox_width, bbox_height):
    cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                int(round(bbox_top - 0.15 * bbox_height)),
                                                                int(round(bbox_width * 1.3)),
                                                                int(round(bbox_height * 1.3)))
    image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
    image = image.resize([64, 64])

    left_x = np.random.randint(0, image.size[0] - 54 - 1)
    left_y = np.random.randint(0, image.size[1] - 54 - 1)
    slide_image_copy1 = image.crop([left_x, left_y, left_x + 54, left_y + 54])
    right_x = np.random.randint(0, image.size[0] - 54 - 1)
    right_y = np.random.randint(0, image.size[1] - 54 - 1)
    slide_image_copy2 = image.crop([10 - right_x, 10 - right_y, 64 - right_x, 64 - right_y])
    return slide_image_copy1, slide_image_copy2


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def read_and_convert(digit_struct_mat_file, path_to_image_files):
    """
    This function reads the mat file and converts the data to the h5 file.
    :param digit_struct_mat_file: the mat file for reading the ground truth data- bounding boxes and the labels
    :param path_to_image_files: The path for the training data images files
    :return: Images with their labels for other uses in this file- but all is saved as h5 file for other files later.
    """

    numbers_image = len(path_to_image_files)
    global numbers_count
    if numbers_image == numbers_count:
        return 0, 0, 0, 0
    if numbers_count % 100 == 0:
        print(numbers_count)
    path_to_image_file = path_to_image_files[numbers_count]  # visit every picture.
    index = int(path_to_image_file.split('\\')[-1].split('.')[0]) - 1  # !! '\\' windows env and '/' Linux env
    numbers_count += 1

    # read the .mat data
    attrs_dict = {}  # Extract the contents in .mat file.
    f = digit_struct_mat_file  # h5py object
    item = f['digitStruct']['bbox'][index].item()
    cur_name_in_h5 = get_name(index=index, hdf5_data=f)

    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = f[item][key]
        values = [f[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs_dict[key] = values

    attrs = attrs_dict
    label_of_digits = attrs['label']

    cur_img = cv2.imread(DATA_DIR + '/train/' + cur_name_in_h5, 0)
    #isolating each digit:
    for i in range(len(label_of_digits)):
        cur_label = int(label_of_digits[i])
        if cur_label == 10:
            #fixing weird format in original labeled data- 0 is tagged as 10
            cur_label = 0
        cur_top = int(attrs_dict['top'][i])
        cur_left = int(attrs_dict['left'][i])
        cur_width = int(attrs_dict['width'][i])
        cur_height = int(attrs_dict['height'][i])
        cur_cropped_digit = cur_img[cur_top : cur_top + cur_height , cur_left : cur_left + cur_width]
        if cur_cropped_digit.shape[0] < CROP_SIZE or cur_cropped_digit.shape[1] < CROP_SIZE:
            continue
        cur_cropped_digit = cv2.resize(cur_cropped_digit, (CROP_SIZE, CROP_SIZE))
        append_to_h5_file(table_name=DIGIT_PATCHES_DIR + "class_" + str(cur_label), cur_label=cur_label, cur_patch=cur_cropped_digit)

    length = len(label_of_digits)
    if length > 5:  # If length of label over 5
        # skip this example
        return read_and_convert(digit_struct_mat_file, path_to_image_files)
    digits = [10, 10, 10, 10, 10]  # digit 10 represents no digit
    for idx_evpicture, label_of_digit in enumerate(label_of_digits):
        digits[idx_evpicture] = int(label_of_digit if label_of_digit != 10 else 0)  # label 10 is essentially digit zero

    attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x],
                                                           [attrs['left'], attrs['top'], attrs['width'],
                                                            attrs['height']])
    min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                min(attrs_top),
                                                max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                max(map(lambda x, y: x + y, attrs_top, attrs_height)))
    center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                    (min_top + max_bottom) / 2.0,
                                    max(max_right - min_left, max_bottom - min_top))
    bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                    center_y - max_side / 2.0,
                                                    max_side,
                                                    max_side)
    slide_image1, slide_image2 = step_process(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width,
                                              bbox_height)

    return slide_image1, slide_image2, length, digits


def convert_to_reformat(path_to_dataset_dir_and_digit_struct_mat_file_tuples, Flag):
    num_examples = []
    count_train = 0
    count_val = 0
    count_test = 0
    other_image = []
    images_examples = {}
    characters = ['length', 'image', 'labels']
    if Flag == True:
        for i in range(2):
            images_examples[i] = {}
            for indej in characters:
                images_examples[i][indej] = []
    else:
        images_examples[0] = {}
        for indej in characters:
            images_examples[0][indej] = []
    for path_to_dataset_dir, path_to_digit_struct_mat_file in path_to_dataset_dir_and_digit_struct_mat_file_tuples:
        path_to_image_files = []
        global numbers_count
        numbers_count = 0
        print(path_to_dataset_dir)
        file_glob = os.path.join(path_to_dataset_dir, '*.png')
        print(file_glob)
        path_to_image_files.extend(glob.glob(file_glob))
        total_files = len(path_to_image_files)
        print('%d files found in %s' % (total_files, path_to_dataset_dir))

        with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
            for index, path_to_image_file in enumerate(path_to_image_files):
                # print('(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file))
                example1, example2, length, digits = read_and_convert(digit_struct_mat_file, path_to_image_files)
                if length == 0:
                    break
                else:
                    if Flag == True:
                        if random.random() > 0.1:
                            id = 0
                            for i in characters:
                                images_examples[id][i].append(0)
                            image_array1 = np.asanyarray(example1, 'float32')
                            images_examples[id]['image'][count_train] = image_array1
                            images_examples[id]['length'][count_train] = length
                            images_examples[id]['labels'][count_train] = digits
                            count_train += 1
                        else:
                            id = 1
                            for i in characters:
                                images_examples[id][i].append(0)
                            image_array1 = np.asanyarray(example1, 'float32')
                            images_examples[id]['image'][count_val] = image_array1
                            images_examples[id]['length'][count_val] = length
                            images_examples[id]['labels'][count_val] = digits
                            count_val += 1
                    else:
                        id = 0
                        for i in characters:
                            images_examples[id][i].append(0)
                        image_array1 = np.asanyarray(example1, 'float32')
                        images_examples[id]['image'][count_test] = image_array1
                        images_examples[id]['length'][count_test] = length
                        images_examples[id]['labels'][count_test] = digits
    return images_examples


def main():
    path_to_train_dir = os.path.join(DATA_DIR, 'train')
    path_to_test_dir = os.path.join(DATA_DIR, 'test')
    path_to_train_digit_struct_mat_file = os.path.join(path_to_train_dir, 'digitStruct.mat')
    path_to_test_digit_struct_mat_file = os.path.join(path_to_test_dir, 'digitStruct.mat')

    path_to_h5_train_file = os.path.join(DATA_DIR, 'train_set.h5')
    path_to_h5_test_file = os.path.join(DATA_DIR, 'test_set.h5')
    path_to_h5_val_file = os.path.join(DATA_DIR, 'val_set.h5')
    train_flag = True
    test_flag = False
    val_flag = False
    if os.path.exists(path_to_h5_train_file):
        print('The file %s already exists' % path_to_h5_train_file)
        # train_flag = False
    if os.path.exists(path_to_h5_test_file):
        print('The file %s already exists' % path_to_h5_test_file)
        test_flag = False
    if os.path.exists(path_to_h5_val_file):
        print('The file %s already exists' % path_to_h5_val_file)
        val_flag = False
    print('Processing train and val data')
    if train_flag == True:
        train_val_set = convert_to_reformat([(path_to_train_dir, path_to_train_digit_struct_mat_file)], True)
    print('Processing test data')
    if test_flag == True:
        test_set = convert_to_reformat([(path_to_test_dir, path_to_test_digit_struct_mat_file)], False)
        file = h5py.File(path_to_h5_test_file, 'w')
    print('Processing val data')
    if val_flag == True:
        # val_set,val_labels,val_length= convert_to_reformat([(path_to_val_dir, path_to_val_digit_struct_mat_file)])
        file = h5py.File(path_to_h5_val_file, 'w')
        file.create_dataset('val_set', data=np.array(train_val_set[1]['image']))
        file.create_dataset('val_labels', data=np.array(train_val_set[1]['labels']))
        file.create_dataset('val_length', data=np.array(train_val_set[1]['length']))
        file.close()

    print('Done!')


if __name__ == '__main__':
    main()

