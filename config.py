# Image cropping settings
ORIGINAL_WIDTH = 4000
ORIGINAL_HEIGHT = 3000
PAD = 25
ZOOM = False
CUT_SIZE = 144  # 145                                # Image patch size
GT_SIZE = 144   # 72                                 # Ground truth size
ADAPT_HIST_EQUAL = False                       # ADAPTIVE HISTOGRAM EQUALIZATION
RGB_GRAYSCALE = 'RGB'
GRID_SIZE = 400
CLIP_LIMIT = 2.0
DOWNSAMPLING = 0

IMG_RESOLUTION = 1.58       # CM PER PIXEL (BASED ON GSD)

# Inputs
TRAIN_INPUT = './datasets/train_all/'
VALIDATION_INPUT = './datasets/validation_all/'
LABEL_INPUT = './datasets/labels_all/'
LABEL_VALIDATION_INPUT = './datasets/labels_all/'

# Outputs
TRAIN_OUTPUT = '../Train/'
VALIDATION_OUTPUT = '../Validation/'
DOWNSAMPLING_OUTPUT = '../Downsampling/'
TRAIN_OUTPUT_IMAGE = '../Train/images/'
TRAIN_LABEL_OUTPUT = '../Train/labels/'
VALIDATION_OUTPUT_IMAGE = '../Validation/images/'
VALIDATION_LABEL_OUTPUT = '../Validation/labels/'
DOWNSAMPLING_OUTPUT_IMAGE = '../Downsampling/images/'
DOWNSAMPLING_LABEL_OUTPUT = '../Downsampling/labels/'


TEMP_OUTPUT = './Results/temp_test/'

