


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from keras.optimizers import SGD, RMSprop, Adam, Nadam, Adagrad, Adadelta, Adamax

from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import AveragePooling2D, MaxPooling2D
from keras.layers import Input,Conv2D
from keras.models import Model
from keras.utils.vis_utils import plot_model

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121
import efficientnet.keras as efn

import keras
import numpy as np

def get_imagenet_model(model_name, input_tensor):
    model_names = ['ResNet50', 'VGG16', 'VGG19', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121',
                   'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                   'EfficientNetB5',
                   'EfficientNetB6', 'EfficientNetB7', 'EfficientNetL2']
    if model_name not in model_names:
        print('ERROR - Undefined Model!')
        sys.exit()
    if model_name == 'ResNet50':
        pre_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'VGG16':
        pre_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'VGG19':
        pre_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'InceptionV3':
        pre_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'InceptionResNetV2':
        pre_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'DenseNet121':
        pre_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetB0':
        pre_model = efn.EfficientNetB0(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetB1':
        pre_model = efn.EfficientNetB1(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetB2':
        pre_model = efn.EfficientNetB2(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetB3':
        pre_model = efn.EfficientNetB3(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetB4':
        pre_model = efn.EfficientNetB4(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetB5':
        pre_model = efn.EfficientNetB5(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetB6':
        pre_model = efn.EfficientNetB6(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetB7':
        pre_model = efn.EfficientNetB7(input_tensor=input_tensor, weights='imagenet', include_top=False)
    if model_name == 'EfficientNetL2':
        pre_model = efn.EfficientNetL2(input_tensor=input_tensor, weights='imagenet', include_top=False)

    return pre_model

def load_model(model_name, input_shape=(341, 341, 3),nb_classes=3):
    input_tensor = Input(shape=input_shape)
    input_tensor_a = Input(shape=input_shape)
    input_tensor_b = Input(shape=input_shape)

    pre_model = get_imagenet_model(model_name, input_tensor)
    output_a = pre_model(input_tensor_a)
    output_b = pre_model(input_tensor_b)
    x = keras.layers.concatenate([output_a, output_b])
    x = AveragePooling2D(pool_size=(7, 7))(x)
    print('SHAPE: ', x.shape)
    x = Flatten()(x)
    print('SHAPE: ', x.shape)
    x = Dense(1024, activation='relu')(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=[input_tensor_a, input_tensor_b], outputs=x)
    return model


def main():
    model = load_model('DenseNet121')
    model.summary()

    test_a = np.random.random((1, 341, 341, 3))
    test_b = np.random.random((1, 341, 341, 3))
    test_y = np.random.random((1, 3))

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([test_a, test_b], test_y, batch_size=1, epochs=1, verbose=1)


if __name__ == '__main__':
    main()