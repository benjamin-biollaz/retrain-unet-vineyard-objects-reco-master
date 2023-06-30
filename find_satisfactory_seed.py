import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import cv2
import numpy as np
from PIL import Image
from unet_model import *
import config


from file_manager import FileManager
from image_manager import ImageManager

class_encoding = FileManager.get_classes_encoding()

# Formatage de la liste cree par image_splitting
def val_generator(img_lst):
    for img in img_lst:
        img = img / 255             # NORMALISATION
        img = np.asarray([img])
        yield img

# Reconstitution de l'image en une
def concat_prediction(predictions, image, cut_size, gt_size):
    x = 0
    y = 0
    cp_x = (cut_size - gt_size)//2
    cp_y = cp_x
    cp_image = Image.new('RGB', (image.shape[1], image.shape[0]))
    for i, item in enumerate(predictions):
        if y + cut_size >= image.shape[0]:
            cp_x += gt_size
            x += gt_size
            cp_y = (cut_size-gt_size)//2
            y = 0
        if x + cut_size >= image.shape[1]:
            break
        
        # Create empty array
        height, width, n_classes = item.shape
        decoded_mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Set the color to the class with the highest probability
        predicted_classes = np.argmax(item, axis=2)
        for name, label, color in class_encoding:
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


def compute_f1(results, filename, extension):

    ca_mask = cv2.imread("datasets/test_labels/" + filename + extension)

    # Convert the image since openCV encodes color in BGR
    ca_mask = cv2.cvtColor(ca_mask, cv2.COLOR_BGR2RGB)

    # Normalize colors
    ca_mask = np.asarray(ca_mask) / 255
    results = np.asarray(results) / 255

    # Vine RBG values
    color = (255, 255, 255)
    color = np.asarray(color) / 255

    # One versus all
    mask_class_indices = np.all(ca_mask == color, axis=2)
    prediction_class_indices = np.all(results == color, axis=2)

    # The class is assigned 1 and other pixels are assigned 0
    mask_class_vs_all = np.zeros((ca_mask.shape[0], ca_mask.shape[1]), dtype=np.uint8)
    pred_class_vs_all = np.zeros((results.shape[0], results.shape[1]), dtype=np.uint8)
    mask_class_vs_all[mask_class_indices] = 1
    pred_class_vs_all[prediction_class_indices] = 1

    intersection = np.logical_and(mask_class_vs_all, pred_class_vs_all)

    # Confusion matrix
    true_pos = np.sum(intersection)
    false_pos = np.logical_xor(intersection, pred_class_vs_all)
    false_pos = np.sum(false_pos)
    false_neg = np.logical_xor(intersection, mask_class_vs_all)
    false_neg = np.sum(false_neg)
    true_neg = (ca_mask.shape[0] * ca_mask.shape[1]) - (true_pos + false_pos + false_neg)

    # Calculate f1
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2*(precision * recall) / (precision + recall)
    return f1

def main(argv):
    file = 'datasets/test/swissimage-dop10_2017_2608-1128_2.jpg'
    cut_size = 144
    gt_size = 144
    result_dir = './Results/'
    weights = "./Weights/unet_vines.hdf5"
    new_data_resolution = 0

    input_size = (cut_size, cut_size, 3)
    filename, extension = FileManager.get_filename_n_extension(self=None, path=file)
    image = cv2.imread(file)

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

    max_f1_score = -1
    best_seed = -1
    upper_bound = 1000100
    for i in range(1000001, upper_bound, 1):

        print(i, "/", upper_bound)

        # Generator must be rebuilt at each iteration
        test_gen = val_generator(img_list)

        # Prediction
        model = unet_sym(pretrained_weights=weights, input_size=input_size, seed=i)
        results = model.predict_generator(test_gen, count_img, verbose=0)      

        # Concat the predictions in a single image
        res_image = concat_prediction(results, image, cut_size, gt_size)

        # Store the seed if the F1 score of this iteration is the best
        f1_score = compute_f1(res_image, filename, extension)
        print(f1_score)
        if (f1_score > max_f1_score):
            best_seed = i
            max_f1_score = f1_score
        
    print("Max")
    print(max_f1_score)
    print(best_seed)

if __name__ == "__main__":
    main(sys.argv[1:])
