import shutil
import os

def remove_patches(path_to_patches):
    shutil.rmtree(path_to_patches)
    os.mkdir(path_to_patches)


def main():
    # data augmentation
    remove_patches("datasets/data_augmentation/images")
    remove_patches("datasets/data_augmentation_labels/labels")

    # validation
    remove_patches("datasets/validation/images")
    remove_patches("datasets/validation_labels/labels")

    # train
    remove_patches("datasets/train/images")
    remove_patches("datasets/train_labels/labels")


main()