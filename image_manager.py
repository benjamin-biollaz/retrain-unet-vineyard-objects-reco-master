import cv2
import config
import tensorflow as tf
import numpy as np

from file_manager import FileManager

class ImageManager:
    def __init__(self, cut_size, gt_size):
       self.cut_size = cut_size
       self.gt_size = gt_size
       self.fileManager = FileManager()

    
    ## Formate image according to algorithm input
    def image_splitting(self, image):
        is_tab = list()
        is_x = 0
        pad = int((config.CUT_SIZE - self.cut_size) / 2)
        pad_left_top = pad

        if pad * 2 < config.CUT_SIZE:
            pad_left_top = pad + 1

        while is_x + self.cut_size < image.shape[1]:
            is_y = 0
            while is_y + self.cut_size < image.shape[0]:
                is_tab.append(
                    cv2.resize(
                        image[is_y : is_y + self.cut_size, is_x : is_x + self.cut_size],
                        (config.CUT_SIZE, config.CUT_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                # is_tab.append(image[is_y:is_y + cut_size, is_x: is_x + cut_size])
                # is_tab.append(cv2.copyMakeBorder(image[is_y:is_y + cut_size, is_x: is_x + cut_size],
                #                                  pad_left_top, pad, pad_left_top, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]))
                is_y += self.gt_size
            is_x += self.gt_size
        return is_tab


    ## Create patches according to the input size
    def create_patches(
        self,
        files_path,
        files_subfolder,
        new_data_resolution,
        pretrained_resolution,
        retrain_with_initial_ratio,
        retrain_with_new_ratio
    ):
        print('Creating patches')

        sample_paths = self.fileManager.get_sample(files_path, files_subfolder)

        for path in sample_paths:
            file_name = self.fileManager.get_filename_n_extension(path)[0]
            file_extension = self.fileManager.get_filename_n_extension(path)[1]
            img = cv2.imread(path)

            if retrain_with_initial_ratio == True and retrain_with_new_ratio == False:
                if img.shape[0] < 3000:
                    img_height = img.shape[0]
                    ratio = (config.ORIGINAL_HEIGHT / img_height) + 2
                    cut_size = int(config.CUT_SIZE / ratio)
                    gt_size = int(config.GT_SIZE / ratio)

            if retrain_with_new_ratio == True:
                ratio = new_data_resolution / pretrained_resolution
                cut_size = int(config.CUT_SIZE / ratio)
                gt_size = int(config.GT_SIZE / ratio)

            patches = self.image_splitting(img)
            # print(len(patches))
            i = 1
            for patch in patches:
                cv2.imwrite(
                    files_path
                    + files_subfolder
                    + "/"
                    + file_name
                    + "_patch_"
                    + str(i)
                    + file_extension,
                    patch,
                )
                i += 1
                # print(len(patch))


    ## Pre-processing for mask and images
    def data_generator(self, images_path, images_subfolder, labels_path, labels_subfolder, batch_size):

        image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
        mask_generator = tf.keras.preprocessing.image.ImageDataGenerator()

        print('Generating images')

        images = image_generator.flow_from_directory(
            directory=images_path,
            classes=[images_subfolder],
            class_mode=None,
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
            class_mode=None,
            color_mode="grayscale",
            target_size=(self.gt_size, self.gt_size),
            batch_size=batch_size,
            shuffle=False,
            seed=42,
        )

        zip_set = zip(images, masks)

        for img, msk in zip_set:
            if np.max(img) > 1:
                img = img / 255
                msk = msk / 255
                msk[msk >= 0.5] = int(1)
                msk[msk < 0.5] = int(0)
                msk = msk.astype(np.bool)
                label = np.zeros(
                    (msk.shape[0], (self.gt_size, self.gt_size)[0], (self.gt_size, self.gt_size)[0], 2),
                    dtype=np.bool,
                )
                label[:, :, :, 0] = msk[:, :, :, 0] == 1
                label[:, :, :, 1] = msk[:, :, :, 0] == 0
            yield img, label


## Augment images trough roation and flip
    def augment_data(self, path, subfolder, augmentation_path):
        
        print('Augmenting images')

        images_paths = self.fileManager.get_sample(path, subfolder)

        for path in images_paths:
            file_name = self.fileManager.get_filename_n_extension(path)[0]
            file_extension = self.fileManager.get_filename_n_extension(path)[1]
            img = cv2.imread(path)

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
