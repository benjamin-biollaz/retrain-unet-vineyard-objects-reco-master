import cv2
import numpy as np

path_file = 'Results/comparaison_meilleur/swissimage_2020_patch-test/ite-11-01_swissimage-dop10_2020_2582-1115_3_patch-test_22_2220210710-174717.png'
path_mask = 'datasets/test_labels/swissimage-dop10_2020_2582-1115_3_patch-test.png'

set_threshold = True
threshold = 60      # Value between 0 and 255

def get_stats(results, ca_mask):
    if set_threshold == True:
        grayImage = cv2.cvtColor(results, cv2.COLOR_BGR2GRAY)
        (thresh, results) = cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY)

    ca_mask = np.asarray(ca_mask) / 255
    results = np.asarray(results) / 255
    ca_mask = ca_mask[:, :, 0]
    if set_threshold == False:
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


image = cv2.imread(path_file)
mask = cv2.imread(path_mask)
get_stats(image, mask)