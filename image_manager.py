import cv2
import config
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

from file_manager import FileManager


class ImageManager:
    def __init__(self, cut_size, gt_size):
       self.cut_size = cut_size
       self.gt_size = gt_size
       self.fileManager = FileManager()

    def image_splitting(self, image, cut_size, gt_size):
        is_tab = list()
        is_x = 0
        pad = int((config.CUT_SIZE - cut_size) / 2)
        pad_left_top = pad

        if pad*2 < config.CUT_SIZE:
            pad_left_top = pad+1

        while is_x + cut_size < image.shape[1]:  # IMAGE.SHAPE[1] = LARGEUR
            is_y = 0
            while is_y + cut_size < image.shape[0]: # IMAGE.SHAPE[0] = HAUTEUR
                is_tab.append(cv2.resize(image[is_y:is_y + cut_size, is_x: is_x + cut_size],(config.CUT_SIZE, config.CUT_SIZE),interpolation=cv2.INTER_AREA))
                # is_tab.append(image[is_y:is_y + cut_size, is_x: is_x + cut_size])
                # is_tab.append(cv2.copyMakeBorder(image[is_y:is_y + cut_size, is_x: is_x + cut_size],
                #                                  pad_left_top, pad, pad_left_top, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]))
                is_y += gt_size
            is_x += gt_size
        return is_tab

    # Create patches according to the input size
    def create_patches(self, files_path, files_subfolder, pretrained_resolution, new_data_resolution,  retrain_with_initial_ratio):
        sample_paths = self.fileManager.get_sample(files_path, files_subfolder)

        # Each path points to a full-size image
        for path in sample_paths:
            file_name = self.fileManager.get_filename_n_extension(path)[0]
            file_extension = self.fileManager.get_filename_n_extension(path)[1]
            img = cv2.imread(path)

            if retrain_with_initial_ratio:
                if img.shape[0] < 3000:
                    img_height = img.shape[0]
                    ratio = (config.ORIGINAL_HEIGHT / img_height) + 2
                    cut_size_with_ratio = int(config.CUT_SIZE / ratio)
                    gt_size_with_ratio = int(config.GT_SIZE / ratio)

            else: 
                ratio = new_data_resolution/pretrained_resolution
                cut_size_with_ratio = int(config.CUT_SIZE / ratio)
                gt_size_with_ratio = int(config.GT_SIZE / ratio)

            patches = self.image_splitting(img, cut_size_with_ratio, gt_size_with_ratio)
            # print(len(patches))
            i = 1
            for patch in patches:
                cv2.imwrite(files_path + files_subfolder + '/' + file_name + '_patch_' + str(i) + file_extension, patch)
                i += 1
                # print(len(patch))

    # Pre-processing for mask and images
    def data_generator(self, images_path, images_subfolder, labels_path, labels_subfolder, batch_size):

        image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
        mask_generator = tf.keras.preprocessing.image.ImageDataGenerator()

        print('Generating images')
        images = image_generator.flow_from_directory(
            directory=images_path,
            classes=[images_subfolder],
           # class_mode=None,
            color_mode="rgb",
            target_size=(self.cut_size, self.cut_size),
            batch_size=batch_size,
            shuffle=False,
            seed=42,
        )

        print('Generating masks')
        masks = mask_generator.flow_from_directory(
            directory=labels_path,
            classes=[labels_subfolder],
           # class_mode=None,
            color_mode="rgb",
            target_size=(self.gt_size, self.gt_size),
            batch_size=batch_size,
            shuffle=False,
            seed=42,
        )

        palette = {
            0 : (255,  255, 255), # White = vine line
            1 : (0.0,  0.0,  0.0), # Black = other / background
            2 : (215,  14, 50), # Red = roofs
        }
        
        zip_set = zip(images, masks)
      
        for img, msk in zip_set:
            batch_size, height, width, rgb = msk[0].shape

            # Creating a 4-dim array of dimensions [batch size, height, width, number of classes]
            label = np.zeros((msk[0].shape[0], height, width, len(palette)), dtype=np.uint8)

            # A batch contains several masks instances
            for individual_mask_index in range(batch_size):
                # One hot encoding rgb values
                label[individual_mask_index, :, :, :] = self.encode_mask(msk[0][individual_mask_index], palette)
            yield img, label
      
    def encode_mask(self, mask, palette):
        height, width, rgb = mask.shape
        encoded_mask = np.zeros((height, width, len(palette)), dtype=np.uint8)

        for label, color in palette.items():
            #indexes = mask[:,:] == color
            #indexes = np.all(mask == color, axis=2)
            indexes = np.all(np.abs(mask - color) <= 10, axis=2)
            encoded_mask[indexes] = to_categorical([label], len(palette), dtype ="uint8")
            
            # indexes1 = np.all(encoded_mask == [0,1,0])
            # if (label == 1 and len(encoded_mask[indexes1]) != 0):
            #     print(encoded_mask[indexes1])

        # pixels with no class are categorised as background
        indexes = np.all(encoded_mask == [0, 0, 0], axis=2)
        encoded_mask[indexes] = to_categorical(1, len(palette), dtype ="uint8")

        return encoded_mask

    # Augment images trough roation and flip
    def augment_data(self, path, subfolder, augmentation_path):
        images_paths = self.fileManager.get_sample(path, subfolder)

        for path in images_paths:
            file_name = self.fileManager.get_filename_n_extension(path)[0]
            file_extension = self.fileManager.get_filename_n_extension(path)[1]
            img = cv2.imread(path)

            # Save untempered image
            cv2.imwrite(
                augmentation_path + file_name + file_extension,
                img,
            )

            # Rotation by 90°
            transformation = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(
                augmentation_path + file_name + "_aug-rot-90_" + file_extension,
                transformation,
            )

            # Rotation by 180°
            transformation = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(
                augmentation_path + file_name + "_aug-rot-180_" + file_extension,
                transformation,
            )

            # Rotation by 270°
            transformation = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(
                augmentation_path + file_name + "_aug-rot-270_" + file_extension,
                transformation,
            )

            # Vertical flip
            transformation = cv2.flip(img, 0)
            cv2.imwrite(
                augmentation_path + file_name + "_aug-flip-ver_" + file_extension,
                transformation,
            )

            # horizontal flip
            transformation = cv2.flip(img, 1)
            cv2.imwrite(
                augmentation_path + file_name + "_aug-flip-hor_" + file_extension,
                transformation,
            )