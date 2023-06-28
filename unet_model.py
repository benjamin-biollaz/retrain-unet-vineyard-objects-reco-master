# ********************************************************************************************************* #
#  ____            _            _          _                                                                #
# |  _ \ _   _  __| | ___      | |    __ _| |__           unet_model_mathematica.py                         #
# | | | | | | |/ _` |/ _ \_____| |   / _` | '_ \          By: Jérôme Treboux <jerome.treboux@hevs.ch>       #
# | |_| | |_| | (_| |  __/_____| |__| (_| | |_) |         Created: 2021/04/30 10:44:14 by Jérôme Treboux    #
# |____/ \__,_|\__,_|\___|     |_____\__,_|_.__/          Updated: 2021/04/30 10:44:14 by Jérôme Treboux    #
#                                                                                                           #
# ********************************************************************************************************* #

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
from tensorflow.keras import initializers
from file_manager import FileManager

class_encoding = FileManager.get_classes_encoding()
nb_classes = len(class_encoding)

# UNET Model : Symetrique
def unet_sym(pretrained_weights=None, input_size=(144, 144, 3)):
    
    initializer = tf.keras.initializers.he_normal()

    inputs = Input(input_size)
    conv1 = Conv2D(3, (2, 2), activation='relu', padding='valid', kernel_initializer=initializer, strides=2)(inputs)
    conv2 = Conv2D(6, (3, 3), activation='relu', padding='valid', kernel_initializer=initializer, strides=3)(conv1)
    conv3 = Conv2D(12, (5, 5), activation='relu', padding='valid', kernel_initializer=initializer)(conv2)
    conv4 = Conv2D(12, (5, 5), activation='relu', padding='valid', kernel_initializer=initializer)(conv3)
    conv5 = Conv2D(18, (5, 5), activation='relu', padding='valid', kernel_initializer=initializer)(conv4)
    conv6 = Conv2D(18, (5, 5), activation='relu', padding='valid', kernel_initializer=initializer)(conv5)
    conv7 = Conv2D(24, (5, 5), activation='relu', padding='valid', kernel_initializer=initializer)(conv6)
    up7 = UpSampling2D(size=(3, 3))(conv7)
    merge7 = concatenate([conv5, up7], axis=3)
    up8 = UpSampling2D(size=(2, 2))(merge7)
    merge8 = concatenate([conv2, up8], axis=3)
    up9 = UpSampling2D(size=(3, 3))(merge8)
    up10 = UpSampling2D(size=(2, 2))(up9)
    
    conv10 = Conv2D(nb_classes, (1, 1), activation='softmax', padding='same', kernel_initializer=initializer)(up10)

    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
     model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)

    return model

    


