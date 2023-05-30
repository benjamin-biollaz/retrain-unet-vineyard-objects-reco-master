import cv2

def get_binary_img(path, binary_name, threshold):

    path = path
    binary_name = binary_name
    threshold = threshold

    # Get the image
    originalImage = cv2.imread(path)

    # Convert to grayscale
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    # Convert to binary
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY)

    # Save the binary image
    cv2.imwrite('Results/binary/' + binary_name + '_THR-' + str(threshold) + '_BINARY.png', blackAndWhiteImage)

for thr in [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225]:

    # BENUNE_4_DJ0012
    get_binary_img(path='Results/comparaison_meilleur/BENUNE_4_DJ0012/ite-00_BENUNE_4_DJ0012_144_14420210530-173831.jpg',
               binary_name='ite-00_BENUNE_4_DJ0012',
               threshold=thr)

    get_binary_img(path='Results/comparaison_meilleur/BENUNE_4_DJ0012/ite-11-01_BENUNE_4_DJ0012_144_14420210708-133939.jpg',
               binary_name='ite-11-01_BENUNE_4_DJ0012',
               threshold=thr)

    # Bernunes_Google_1_patch-test
    get_binary_img(path='Results/comparaison_meilleur/Bernunes_Google_1_patch-test/ite-00-01_Bernunes_Google_1_patch-test_20_2020210710-184250.PNG',
               binary_name='ite-00-01_Bernunes_Google_1_patch-test',
               threshold=thr)

    get_binary_img(path='Results/comparaison_meilleur/Bernunes_Google_1_patch-test/ite-11-01_Bernunes_Google_1_patch-test_28_2820210710-192709.PNG',
               binary_name='ite-11-01_Bernunes_Google_1_patch-test',
               threshold=thr)

    # DJI_0035
    get_binary_img(path='Results/comparaison_meilleur/DJI_0035/ite-00_DJI_0035_144_14420210530-173914.jpg',
               binary_name='ite-00_DJI_0035',
               threshold=thr)

    get_binary_img(path='Results/comparaison_meilleur/DJI_0035/ite-11-01_DJI_0035_144_14420210708-134033.jpg',
               binary_name='ite-11-01_DJI_0035',
               threshold=thr)

    # swissimage_2017_patch-test
    get_binary_img(path='Results/comparaison_meilleur/swissimage_2017_patch-test/ite-00-01_swissimage-dop10_2017_2608-1128_2_patch-test_17_1720210710-160505.png',
               binary_name='ite-00-01_swissimage-dop10_2017_2608-1128_2_patch-test',
               threshold=thr)

    get_binary_img(path='Results/comparaison_meilleur/swissimage_2017_patch-test/ite-11-01_swissimage-dop10_2017_2608-1128_2_patch-test_22_2220210710-174451.png',
               binary_name='ite-11-01_swissimage-dop10_2017_2608-1128_2_patch-test',
               threshold=thr)

    # swissimage_2020_patch-test
    get_binary_img(path='Results/comparaison_meilleur/swissimage_2020_patch-test/ite_00-01_swissimage-dop10_2020_2582-1115_3_patch-test_17_1720210710-162739.png',
               binary_name='ite_00-01_swissimage-dop10_2020_2582-1115_3_patch-test',
               threshold=thr)

    get_binary_img(path='Results/comparaison_meilleur/swissimage_2020_patch-test/ite-11-01_swissimage-dop10_2020_2582-1115_3_patch-test_22_2220210710-174717.png',
               binary_name='ite-11-01_swissimage-dop10_2020_2582-1115_3_patch-test',
               threshold=thr)

    # VINES_1_patch-test
    get_binary_img(path='Results/comparaison_meilleur/VINES_1_patch-test/ite-00-01_VINES_1_patch-test_17_1720210710-184141.JPG',
               binary_name='ite-00-01_VINES_1_patch-test',
               threshold=thr)

    get_binary_img(path='Results/comparaison_meilleur/VINES_1_patch-test/ite-11-01_VINES_1_patch-test_22_2220210710-192622.JPG',
               binary_name='ite-11-01_VINES_1_patch-test',
               threshold=thr)

    # Bernunes_Google_1_28_2820210708-134249
    get_binary_img(path='Results/iteration_11/Bernunes_Google_1_28_2820210708-134249.PNG',
               binary_name='ite-11-01_Bernunes_Google_1',
               threshold=thr)

    # swissimage-dop10_2017_2608-1128_2_22_2220210708-134507
    get_binary_img(path='Results/iteration_11/swissimage-dop10_2017_2608-1128_2_22_2220210708-134507.png',
               binary_name='ite-11-01_swissimage_2017',
               threshold=thr)

    # swissimage-dop10_2020_2582-1115_3_22_2220210708-134929
    get_binary_img(path='Results/iteration_11/swissimage-dop10_2020_2582-1115_3_22_2220210708-134929.png',
               binary_name='ite-11-01_swissimage_2020',
               threshold=thr)

    # VINES_1_22_2220210708-134145
    get_binary_img(path='Results/iteration_11/VINES_1_22_2220210708-134145.JPG',
               binary_name='ite-11-01_VINES_1',
               threshold=thr)