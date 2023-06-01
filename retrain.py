
### Run the libraries

import os
import sys

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import cv2
import numpy as np
import config
from datetime import datetime
import traceback
from unet_model import unet_sym

### Set the initial variables ------------------------------------------------------------------------------------------

## Model to retrain
cut_size = 144
gt_size = 144
input_size = (cut_size, cut_size, 3)
weights_path = 'Weights/'
pretrained_weights = 'unet_vines.hdf5'
trainable_layers = ['conv2d_7', 'conv2d_6', 'conv2d_5', 'conv2d_4', 'conv2d_3', 'conv2d_2', 'conv2d_1', 'conv2d'] # Choose layers with print model.summary
number_layers_to_retrain = 1        # How many layers are unfrozen
batch_size = 32
epoch = 1
pretrained_resolution = 1.58        # How many cm are covered by a pixel (here GSD)
new_data_resolution = 10            # How many cm are covered by a pixel (here GSD)
retrain_with_initial_ratio = False
retrain_with_new_ratio = True
performance_metric = [tf.keras.metrics.Precision()]  # Which metrics for the retraining
                                                        ## ['accuracy']
                                                        ## [tf.keras.metrics.Precision()]
                                                        ## [tf.keras.metrics.Recall()]

## Datasets
datasets_folder = 'datasets'
subfolder = 'images'
labels_subfolder = 'labels'

# Train dataset
print_train_set = True
train_folder = 'train'
train_labels_folder = 'train_labels'

# Validation dataset
print_validation_set = True
validation_folder = 'validation'
validation_labels_folder = 'validation_labels'

# Data augmentation settings
use_augmentation = True
augmentation_folder = 'data_augmentation'
augmentation_labels_folder = 'data_augmentation_labels'

### Get the images sample paths ----------------------------------------------------------------------------------------

## Get the sample from path

def get_sample(path, subfolder):
    sample_paths = sorted(
        [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if not fname.startswith('.') and fname.find(subfolder)
        ]
    )
    return sample_paths

## Print sample size and paths

def print_set(images_path, images_subfolder, labels_path, labels_subfolder):
    images_sample_paths = get_sample(images_path, images_subfolder)
    labels_sample_paths = get_sample(labels_path, labels_subfolder)

    print('Number of samples:', len(images_sample_paths))
    print('10 first samples : ')

    for image_path, label_path in zip(images_sample_paths[:10], labels_sample_paths[:10]):
        print(image_path, '|', label_path)

## Get the file name and file extension

def get_filename_n_extension(path):
    gfe_split_name = path.split(os.path.sep)
    gfe_file = gfe_split_name[-1]
    gfe_filename = gfe_file.split('.')[0].split('/')[-1]
    gfe_ext = '.' + gfe_file.split('.')[1]
    return gfe_filename, gfe_ext

## Remove existing patches

def remove_patches(files_path, files_subfolder):
    sample_paths = get_sample(files_path + files_subfolder + '/', 'None')
    for path in sample_paths:
        os.remove(path)

    print('Successful patches removal from ' + files_path + files_subfolder + ' !')

### Create datasets ---------------------------------------------------------------------------------------------------

## Formate image according to algorithm input

def image_splitting(image, cut_size, gt_size):
    is_tab = list()
    is_x = 0
    pad = int((config.CUT_SIZE - cut_size) / 2)
    pad_left_top = pad

    if pad*2 < config.CUT_SIZE:
        pad_left_top = pad+1

    while is_x + cut_size < image.shape[1]:
        is_y = 0
        while is_y + cut_size < image.shape[0]:
            is_tab.append(cv2.resize(image[is_y:is_y + cut_size, is_x: is_x + cut_size],
                                     (config.CUT_SIZE, config.CUT_SIZE),
                                     interpolation=cv2.INTER_AREA))
            # is_tab.append(image[is_y:is_y + cut_size, is_x: is_x + cut_size])
            # is_tab.append(cv2.copyMakeBorder(image[is_y:is_y + cut_size, is_x: is_x + cut_size],
            #                                  pad_left_top, pad, pad_left_top, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]))
            is_y += gt_size
        is_x += gt_size
    return is_tab

## Create patches according to the input size

def create_patches(files_path, files_subfolder, cut_size, gt_size):
    sample_paths = get_sample(files_path, files_subfolder)

    for path in sample_paths:
        file_name = get_filename_n_extension(path)[0]
        file_extension = get_filename_n_extension(path)[1]
        img = cv2.imread(path)

        if retrain_with_initial_ratio == True and retrain_with_new_ratio == False:
            if img.shape[0] < 3000:
                img_height = img.shape[0]
                ratio = (config.ORIGINAL_HEIGHT / img_height) + 2
                cut_size = int(config.CUT_SIZE / ratio)
                gt_size = int(config.GT_SIZE / ratio)

        if retrain_with_new_ratio == True:
            ratio = new_data_resolution/pretrained_resolution
            cut_size = int(config.CUT_SIZE / ratio)
            gt_size = int(config.GT_SIZE / ratio)

        patches = image_splitting(img, cut_size, gt_size)
        # print(len(patches))
        i = 1
        for patch in patches:
            cv2.imwrite(files_path + files_subfolder + '/' + file_name + '_patch_' + str(i) + file_extension, patch)
            i += 1
            # print(len(patch))

## Build the data generator

def data_generator(images_path, images_subfolder, labels_path, labels_subfolder):

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    mask_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    images = image_generator.flow_from_directory(directory=images_path,
                                                 classes=[images_subfolder],
                                                 class_mode=None,
                                                 color_mode='rgb',
                                                 target_size=(cut_size, cut_size),
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 seed=42)

    masks = mask_generator.flow_from_directory(directory=labels_path,
                                               classes=[labels_subfolder],
                                               class_mode=None,
                                               color_mode='grayscale',
                                               target_size=(gt_size, gt_size),
                                               batch_size=batch_size,
                                               shuffle=False,
                                               seed=42)

    zip_set = zip(images, masks)

    for img, msk in zip_set:
        if np.max(img) > 1:
            img = img / 255
            msk = msk / 255
            msk[msk >= 0.5] = int(1)
            msk[msk < 0.5] = int(0)
            msk = msk.astype(np.bool)
            label = np.zeros((msk.shape[0], (gt_size, gt_size)[0], (gt_size, gt_size)[0], 2), dtype=np.bool)
            label[:, :, :, 0] = (msk[:, :, :, 0] == 1)
            label[:, :, :, 1] = (msk[:, :, :, 0] == 0)
        yield img, label

## Augment your data

def augment_data(path, subfolder, augmentation_path):

    images_paths = get_sample(path, subfolder)

    for path in images_paths:
        file_name = get_filename_n_extension(path)[0]
        file_extension = get_filename_n_extension(path)[1]
        img = cv2.imread(path)

        # Rotation by 90°
        transformation = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(augmentation_path + file_name + '_aug-rot-90_' + file_extension, transformation)

        # Rotation by 180°
        transformation = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(augmentation_path + file_name + '_aug-rot-180_' + file_extension, transformation)

        # Rotation by 270°
        transformation = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(augmentation_path + file_name + '_aug-rot-270_' + file_extension, transformation)

        # Vertical flip
        transformation = cv2.flip(img, 0)
        cv2.imwrite(augmentation_path + file_name + '_aug-flip-ver_' + file_extension, transformation)

        # horizontal flip
        transformation = cv2.flip(img, 1)
        cv2.imwrite(augmentation_path + file_name + '_aug-flip-hor_' + file_extension, transformation)

### Main function ------------------------------------------------------------------------------------------------------

def main():
    try:

        # sys.stdout = open('retrain/retrain_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt', 'w')

        # Print the number of samples and the 10 first samples
        if print_train_set == True:
            train_images_path = datasets_folder + '/' + train_folder + '/'
            train_labels_path = datasets_folder + '/' + train_labels_folder + '/'
            print('training set')
            print_set(train_images_path, subfolder, train_labels_path, labels_subfolder)

        if print_validation_set == True:
            validation_images_path = datasets_folder + '/' + validation_folder + '/'
            validation_labels_path = datasets_folder + '/' + validation_labels_folder + '/'
            print('validation set')
            print_set(validation_images_path, subfolder, validation_labels_path, labels_subfolder)

        # Load model without pretrained weights
        # no_weigths_model = unet_sym(input_size=input_size)
        # print(no_weigths_model.summary())
        # for layer in no_weigths_model.layers:
        #     print(layer.name,
        #           ' | weights:', len(layer.weights),
        #           ' | trainable weights:', len(layer.trainable_weights),
        #           ' | non trainable weights:', len(layer.non_trainable_weights),
        #           ' | trainable layer:', layer.trainable)
        #     print(layer.weights)

        # Load model with pretrained weights
        load_weights = weights_path + pretrained_weights
        print(load_weights)
        initial_model = unet_sym(pretrained_weights=load_weights, input_size=input_size)
        # print(initial_model.summary())
        # for layer in initial_model.layers:
        #     print(layer.name,
        #           ' | weights:', len(layer.weights),
        #           ' | trainable weights:', len(layer.trainable_weights),
        #           ' | non trainable weights:', len(layer.non_trainable_weights),
        #           ' | trainable layer:', layer.trainable)
        #     print(layer.weights)

        # Freeze the loaded model with pretrained weights
        set_trainable = False
        layers_to_retrain = trainable_layers[0:number_layers_to_retrain]
        for layer in initial_model.layers:
            if layer.name in layers_to_retrain:
                set_trainable = True
            else:
                layer.trainable = False

        # Set the model to retrain
        unet_to_retrain = Model(initial_model.input, initial_model.output)
        print(unet_to_retrain.summary())
        for layer in unet_to_retrain.layers:
            print(layer.name,
                  ' | weights:', len(layer.weights),
                  ' | trainable weights:', len(layer.trainable_weights),
                  ' | non trainable weights:', len(layer.non_trainable_weights),
                  ' | trainable layer:', layer.trainable)
            # print(layer.weights)

        # Compile model
        unet_to_retrain.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=performance_metric)
        #unet_to_retrain.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=performance_metric)


        # Set the images and labels paths
        if use_augmentation == True:
            train_images_path = datasets_folder + '/' + augmentation_folder + '/'
            train_labels_path = datasets_folder + '/' + augmentation_labels_folder + '/'
            augment_data(datasets_folder + '/' + train_folder + '/', subfolder, train_images_path)
            augment_data(datasets_folder + '/' + train_labels_folder + '/', labels_subfolder, train_labels_path)
            print('training set with augmentation')
            print_set(train_images_path, subfolder, train_labels_path, labels_subfolder)
        else:
            train_images_path = datasets_folder + '/' + train_folder + '/'
            train_labels_path = datasets_folder + '/' + train_labels_folder + '/'

        validation_images_path = datasets_folder + '/' + validation_folder + '/'
        validation_labels_path = datasets_folder + '/' + validation_labels_folder + '/'

        # Remove exiting patches for the images and the labels
        remove_patches(datasets_folder + '/' + augmentation_folder + '/', subfolder) # from data_augmentation
        remove_patches(datasets_folder + '/' + augmentation_labels_folder + '/', labels_subfolder) # from data_augmentation_labels
        remove_patches(datasets_folder + '/' + train_folder + '/', subfolder) # from train
        remove_patches(datasets_folder + '/' + train_labels_folder + '/', labels_subfolder) # from train_labels
        remove_patches(validation_images_path, subfolder) # from validation
        remove_patches(validation_labels_path, labels_subfolder) # from validation_labels

        # Prepare patches for the images and the labels
        create_patches(train_images_path, subfolder, cut_size, gt_size)
        create_patches(train_labels_path, labels_subfolder, cut_size, gt_size)
        create_patches(validation_images_path, subfolder, cut_size, gt_size)
        create_patches(validation_labels_path, labels_subfolder, cut_size, gt_size)

        # Create the training data generator
        training_generator = data_generator(train_images_path, subfolder,
                                            train_labels_path, labels_subfolder)

        # Create the validation data generator
        validation_generator = data_generator(validation_images_path, subfolder,
                                              validation_labels_path, labels_subfolder)

        # Get the number of full training sample
        if use_augmentation == True:
            full_training_path = datasets_folder + '/' + augmentation_folder + '/' + subfolder + '/'
        else:
            full_training_path = datasets_folder + '/' + train_folder + '/' + subfolder + '/'
        sample_size = len(get_sample(full_training_path, 'None'))
        print('training sample size :', sample_size)

        # Get the number of full validation sample
        full_validation_path = datasets_folder + '/' + validation_folder + '/' + subfolder + '/'
        val_sample_size = len(get_sample(full_validation_path, 'None'))
        print('validation sample size :', val_sample_size)

        # Retrain the model with fit
        unet_to_retrain.fit(training_generator, epochs=epoch, steps_per_epoch=sample_size//batch_size,
                            validation_data=validation_generator, validation_steps=val_sample_size//batch_size)

        # Save the weigths of the retrained model
        iteration_name = weights_path + 'unet_vines_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.hdf5'
        print(iteration_name)
        unet_to_retrain.save_weights(filepath=iteration_name, overwrite=False)

        # Display the final message
        print('Successful retraining!')

    except Exception as e:

        # Display the error message
        print("Training aborted due to the following error:")
        print(e)

        # print traceback
        traceback.print_exc()

### Run the main function ----------------------------------------------------------------------------------------------


main()
