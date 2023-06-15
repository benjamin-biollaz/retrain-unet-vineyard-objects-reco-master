# ********************************************************************************************************* #
#  ____            _            _          _                                                                #
# |  _ \ _   _  __| | ___      | |    __ _| |__           unet_validation.py                                #
# | | | | | | |/ _` |/ _ \_____| |   / _` | '_ \          By: Nabih <nabih.ali@hevs.ch>                     #
# | |_| | |_| | (_| |  __/_____| |__| (_| | |_) |         Created: 2020/04/16 15:24:08 by Nabih             #
# |____/ \__,_|\__,_|\___|     |_____\__,_|_.__/          Updated: 2021/05/04 09:51:44 by Jérôme Treboux    #
#                                                                                                           #
# ********************************************************************************************************* #

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import cv2
import getopt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from unet_model import *
from datetime import datetime
import config
from image_manager import ImageManager


# Recuperation des filenames et file_extension
def get_filename_n_extension(path):
    gfe_split_name = path.split(os.path.sep)
    gfe_file = gfe_split_name[-1]
    gfe_filename = gfe_file.split(".")[0].split('/')[-1]
    gfe_ext = "." + gfe_file.split(".")[1]
    return gfe_filename, gfe_ext

# Formatage de la liste cree par image_splitting
def val_generator(img_lst):
    for img in img_lst:
        img = img / 255             # NORMALISATION
        img = np.asarray([img])
        yield img


# Reconstitution de l'image en une
def concat_prediction(predictions, image, cut_size, gt_size, pred_val=0.75):
    x = 0
    y = 0
    cp_x = (cut_size - gt_size)//2
    cp_y = cp_x
    cp_image = Image.new('RGB', (image.shape[1], image.shape[0]))
    # predictions = predictions >= pred_val
    for i, item in enumerate(predictions):
        if y + cut_size >= image.shape[0]:
            cp_x += gt_size
            x += gt_size
            cp_y = (cut_size-gt_size)//2
            y = 0
        if x + cut_size >= image.shape[1]:
            break
        
        palette = {
            0 : (215,  14, 50), # Red = roofs
            1 : (0.0,  0.0,  0.0), # Black = other / background
            2 : (255,  255, 255), # White = vine line 
        }
        
        # Create empty array
        height, width, n_classes = item.shape
        decoded_mask = np.zeros((height, width, n_classes), dtype=np.uint8)

        # Set the color to the class with the highest probability
        predicted_classes = np.argmax(item, axis=2)
        for label, color in palette.items():
            mask_indices = np.where(predicted_classes == label)
            decoded_mask[mask_indices] = color
       
        cp_tmp = np.uint8(decoded_mask)
        cp_tmp = Image.fromarray(cp_tmp)
        cp_tmp = cp_tmp.resize((gt_size, gt_size))
        test = cv2.cvtColor(np.array(cp_tmp), cv2.COLOR_RGB2BGR)
        path = os.path.join(config.TEMP_OUTPUT, "test_" + str(cp_x) + str(cp_y) + ".jpg")
        cv2.imwrite(path, test)
        cp_image.paste(cp_tmp, (cp_x, cp_y))
        cp_y += gt_size
        y += gt_size
    return cp_image


def calcul_accuracy(results, filename, extension):
    ca_mask = cv2.imread("datasets/test_labels/" + filename + extension)
    ca_mask = np.asarray(ca_mask) / 255
    results = np.asarray(results) / 255
    ca_mask = ca_mask[:, :, 0]
    results = results[:, :, 0]

    intersection = np.logical_and(ca_mask, results)
    union = np.logical_or(ca_mask, results)

    true_pos = np.sum(intersection)

    false_pos = np.logical_xor(intersection, results)
    false_pos = np.sum(false_pos)

    false_neg = np.logical_xor(intersection, ca_mask)
    false_neg = np.sum(false_neg)

    true_neg = (ca_mask.shape[0] * ca_mask.shape[1]) - (true_pos + false_pos + false_neg)

    total = ca_mask.shape[0] * ca_mask.shape[1]

    print("\n-----------------------------------")
    print("TP =", true_pos, "TN =", true_neg)
    print("FP =", false_pos, "FN =", false_neg)
    print("-----------------------------------")
    print("Max pixels =", total)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    print("-----------------------------------")
    print("Pixel Accuracy =", (true_pos + true_neg) / total)
    print("Precision =", precision)
    print("Recall =", recall)
    print("IoU =", np.sum(intersection) / np.sum(union))
    print("F1 score =", 2*(precision * recall) / (precision + recall))


def adaptive_histogram_equalization_rgb(image: [], grid_size=50):
    """ Return an image RGB with the adaptive histogram equalization applied
    """

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab_image)

    clahe_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))

    lab_planes[0] = clahe_image.apply(lab_planes[0])

    lab_image = cv2.merge(lab_planes)
    bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    return bgr_image


##
# # Usage: Usage: ./unet_validation -f <file path>
# --> optional(-c <cut size> -w <weights path> -r <results folder> -p <results percent>))
# # OBLIGATOIRE
# # file path = chemin du fichier a predire
# # OPTIONNEL
# # cut size = format des images apres decoupe
# # weights path = chemin des poids a utiliser pour la prediction
# # results folder = dossier ou sera sauvegarder l'image
# # results percent = pourcentage minimum accepter
##
def main(argv):
    file = ''
    cut_size = 144
    gt_size = 144
    weights = "./Weights/unet_vines.hdf5"
    result_dir = './Results/'
    percent = 0.6
    stats = False
    hist = False
    new_data_resolution = 0
    try:
        opts, args = getopt.getopt(argv, "hf:c:w:r:p:g:", ["file=", "size=", "weights=", "results=", "stats", "cmpx="])
    except getopt.GetoptError:
        print("unet_validation -f <file path>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Usage: ./unet_validation\n"
                  "             -h\t->\twill show you this help\n"
                  "   -f or --file\t->\tis use to set the path of your image \n"
                  "   -c or --size\t->\tis use to set the cut size of the image\n"
                  "-w or --weights\t->\tis use to set a specific pre-trained weights file\n"
                  "-r or --results\t->\tis use to change the default save results folder\n"
                  "             -p\t->\tuseful to change the expected level of accuracy\n"
                  "             -e\t->\tdo the Adaptive Histogram Equalization\n"
                  "        --stats\t->\tthis option will show you some statistics about\n"
                  "\t\t\t\t\tthe recognition ONLY if you have the label of the image\n"
                  "\t\t\t\t\tin the folder \"dataset/labels\"")
            sys.exit(1)
        elif opt in ("-f", "--file"):
            file = str(arg)
        elif opt in ("-c", "--size"):
            cut_size = int(arg)
        elif opt in ("-w", "--weights"):
            weights = str(arg)
        elif opt in ("-r", "--results"):
            result_dir = str(arg)
        elif opt == "-p":
            percent = float(arg)
        elif opt == "--stats":
            stats = True
        elif opt == "-g":
            gt_size = int(arg)
        elif opt == "--cmpx":
            new_data_resolution = float(arg)

    if file == '':
        print("Usage: unet_validation -f <file path> optional(-c <cut size> -g <gt size> -w <weights path> -r <results folder>)")
        sys.exit(2)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    input_size = (cut_size, cut_size, 3)
    filename, extension = get_filename_n_extension(file)
    image = cv2.imread(file)
    if hist:
        image = adaptive_histogram_equalization_rgb(image)

    if new_data_resolution == 0:
        if image.shape[0] < 3000:
            img_height = image.shape[0]
            ratio = (config.ORIGINAL_HEIGHT/img_height)+2
            cut_size = int(config.CUT_SIZE/ratio)
            gt_size = int(config.GT_SIZE/ratio)
    else:
        ratio = new_data_resolution/config.IMG_RESOLUTION
        cut_size = int(config.CUT_SIZE/ratio)
        gt_size = int(config.GT_SIZE/ratio)

    imageManager = ImageManager(cut_size, gt_size)
    img_list = imageManager.image_splitting(image, cut_size, gt_size)
    count_img = len(img_list)
    test_gen = val_generator(img_list)

    # Prediction avec les weights entrainer
    model = unet_sym(pretrained_weights=weights, input_size=input_size)
    print(model.summary())
    results = model.predict_generator(test_gen, count_img, verbose=1)

    # Sauvegarde du resultat
    res_image = concat_prediction(results, image, cut_size, gt_size, percent)
    if result_dir[-1] == '/':
        res_image.save(result_dir + filename + "_" + str(cut_size) + "_" + str(gt_size) + datetime.now().strftime("%Y%m%d-%H%M%S") + extension)
    else:
        res_image.save(result_dir + "/" + filename + "_" + cut_size + "_" + gt_size + datetime.now().strftime("%Y%m%d-%H%M%S") + extension)
    if stats:
        calcul_accuracy(res_image, filename, extension)


if __name__ == "__main__":
    main(sys.argv[1:])
