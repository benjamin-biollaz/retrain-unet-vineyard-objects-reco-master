import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from datetime import datetime
import math
import traceback

# Project files import
from unet_model import unet_sym
from image_manager import ImageManager
from file_manager import FileManager

### Set the initial variables ------------------------------------------------------------------------------------------

## Model to retrain
cut_size = 144 #patches size for the detection
gt_size = 144 # patches size of the labels
input_size = (cut_size, cut_size, 3)
weights_path = "Weights/"
pretrained_weights = "unet_vines.hdf5"
trainable_layers = ["conv2d_7","conv2d_6","conv2d_5","conv2d_4","conv2d_3","conv2d_2","conv2d_1","conv2d",]  
number_layers_to_retrain = 8  # How many layers are unfrozen
batch_size = 32
epoch = 25
pretrained_resolution = 1.58  # How many cm are covered by a pixel (here GSD)
new_data_resolution = 10  # How many cm are covered by a pixel (here GSD)
retrain_with_initial_ratio = False
performance_metric = [tf.keras.metrics.Precision()]  # Which metrics for the retraining

## Datasets
datasets_folder = "datasets"
subfolder = "images"
labels_subfolder = "labels"

# Train dataset
print_train_set = True
train_folder = "train"
train_labels_folder = "train_labels"
train_images_path = datasets_folder + "/" + train_folder + "/"
train_labels_path = datasets_folder + "/" + train_labels_folder + "/"


# Validation dataset
print_validation_set = True
validation_folder = "validation"
validation_labels_folder = "validation_labels"
validation_images_path = datasets_folder + "/" + validation_folder + "/"
validation_labels_path = datasets_folder + "/" + validation_labels_folder + "/"

# Data augmentation settings
use_augmentation = True
augmentation_folder = "data_augmentation"
augmentation_labels_folder = "data_augmentation_labels"

# Project classes
fileManager = FileManager()
imageManager = ImageManager(cut_size, gt_size)

## Print sample size and paths
def print_set(image_path, images_subfolder, labels_path, labels_subfolder):
    images_sample_paths = fileManager.get_sample(image_path, images_subfolder)
    labels_sample_paths = fileManager.get_sample(labels_path, labels_subfolder)

    print("Number of samples:", len(images_sample_paths))
    print("10 first samples : ")

    for image_path, label_path in zip(images_sample_paths[:10], labels_sample_paths[:10]):
        print(image_path, "|", label_path)


def load_model():

    # Load model with pretrained weights
    load_weights = weights_path + pretrained_weights
    print(load_weights)
    initial_model = unet_sym(pretrained_weights=load_weights, input_size=input_size)

    # Freeze the loaded model with pretrained weights
    layers_to_retrain = trainable_layers[0:number_layers_to_retrain]
    for layer in initial_model.layers:
        if layer.name in layers_to_retrain:
            layer.trainable = True
        else:
            layer.trainable = False

    # Set the model to retrain
    unet_to_retrain = Model(initial_model.input, initial_model.output)
    for layer in unet_to_retrain.layers:
        print(layer.name," | weights:",len(layer.weights)," | trainable weights:",len(layer.trainable_weights),
            " | non trainable weights:",len(layer.non_trainable_weights)," | trainable layer:",layer.trainable,)
    return unet_to_retrain


def print_sample_information():
    if print_train_set:
        print("training set")
        print_set(train_images_path, subfolder, train_labels_path, labels_subfolder)

    if print_validation_set:
        print("validation set")
        print_set(validation_images_path, subfolder, validation_labels_path, labels_subfolder)

def replace_patches(validation_images_path, validation_labels_path, train_images_path, train_labels_path):
        # Remove exiting patches for the images and the labels      
        fileManager.remove_patches(datasets_folder + "/" + augmentation_folder + "/", subfolder)
        fileManager.remove_patches(datasets_folder + "/" + augmentation_labels_folder + "/", labels_subfolder)
        fileManager.remove_patches(datasets_folder + "/" + train_folder + "/", subfolder) 
        fileManager.remove_patches(datasets_folder + "/" + train_labels_folder + "/", labels_subfolder)
        fileManager.remove_patches(validation_images_path, subfolder)  
        fileManager.remove_patches(validation_labels_path, labels_subfolder)

        # Prepare patches for the images and the labels
        print('Creating patches')
        imageManager.create_patches(train_images_path, subfolder, pretrained_resolution, new_data_resolution, retrain_with_initial_ratio)
        print("Images patches finished")
        imageManager.create_patches(train_labels_path, labels_subfolder, pretrained_resolution, new_data_resolution, retrain_with_initial_ratio)
        print("Labels patches finished")
        imageManager.create_patches(validation_images_path, subfolder, pretrained_resolution, new_data_resolution, retrain_with_initial_ratio)
        imageManager.create_patches(validation_labels_path, labels_subfolder, pretrained_resolution, new_data_resolution, retrain_with_initial_ratio)

def augment_images():
        # Set paths
        global train_images_path 
        train_images_path = datasets_folder + "/" + augmentation_folder + "/"
        global train_labels_path 
        train_labels_path = datasets_folder + "/" + augmentation_labels_folder + "/"
        
        # Augment data
        print("Augmenting data")
        imageManager.augment_data(datasets_folder + "/" + train_folder + "/", subfolder, train_images_path)
        imageManager.augment_data(datasets_folder + "/" + train_labels_folder + "/",labels_subfolder,train_labels_path,)

def print_patches_sample_information():
        # Get the number of training sample and print it
        full_training_path = (train_images_path + "/" + subfolder + "/")
        sample_size = len(fileManager.get_sample(full_training_path, "None"))
        print("training sample size :", sample_size)

        # Get the number of full validation sample and print it
        full_validation_path = (datasets_folder + "/" + validation_folder + "/" + subfolder + "/")
        val_sample_size = len(fileManager.get_sample(full_validation_path, "None"))
        print("validation sample size :", val_sample_size)

        return sample_size, val_sample_size
   
    
# Main function ------------------------------------------------------------------------------------------------------
def main():
    try:
        unet_to_retrain = load_model()

        # Compile model
        unet_to_retrain.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=performance_metric)

        # Assert that the images folder exist
        fileManager.create_dataset_directories()

        # Handle the data augmentation
        if use_augmentation:
            augment_images()

        # Print the number of samples and the 10 first samples
        print_sample_information()

        # Delete old patches and create new ones
        replace_patches(validation_images_path, validation_labels_path, train_images_path, train_labels_path)

        # Pre-process images and masks
        training_generator = imageManager.data_generator(train_images_path, subfolder, train_labels_path, labels_subfolder, batch_size)
        validation_generator = imageManager.data_generator(validation_images_path, subfolder, validation_labels_path, labels_subfolder, batch_size)

        sample_size, val_sample_size = print_patches_sample_information()    

        # Retrain the model with fit
        unet_to_retrain.fit(training_generator,epochs=epoch,steps_per_epoch= math.ceil(sample_size / batch_size),
            validation_data=validation_generator,validation_steps= math.ceil(val_sample_size / batch_size),)

        # Save the weigths of the retrained model
        iteration_name = (weights_path+ "unet_vines_"+ datetime.now().strftime("%Y%m%d-%H%M%S")+ ".hdf5")
        print(iteration_name)
        unet_to_retrain.save_weights(filepath=iteration_name, overwrite=False)

        print("Successful retraining!")

    except Exception as e:
        # print traceback
        traceback.print_exc()


### Run the main function ----------------------------------------------------------------------------------------------
main()
