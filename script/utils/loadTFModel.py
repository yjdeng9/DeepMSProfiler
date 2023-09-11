import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam, Adagrad, Adadelta, Adamax

from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
# from tensorflow.keras.utils.vis_utils import plot_model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet121
import efficientnet.tfkeras as efn


def load_imagenet_model(model_name, input_shape, nb_classes=3, pool_size=None):
    input_tensor = Input(shape=input_shape)
    if pool_size is None:
        return get_imagenet_model(model_name, input_tensor, nb_classes=nb_classes)
    else:
        pool_tensor = MaxPooling2D(pool_size=(pool_size, pool_size))(input_tensor)
        pre_model = get_imagenet_model(model_name, pool_tensor, nb_classes=nb_classes)
        model = Model(input_tensor, pre_model.output)

    return model


def get_imagenet_model(model_name, input_tensor, nb_classes):
    model_names = ['ResNet50', 'VGG16', 'VGG19', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121',
                   'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                   'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetL2']

    if model_name not in model_names:
        print('ERROR - Undefined Model!')
        sys.exit()

    if model_name == 'ResNet50':
        pre_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'VGG16':
        pre_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model,nb_classes)
    if model_name == 'VGG19':
        pre_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model,nb_classes)
    if model_name == 'InceptionV3':
        pre_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model,nb_classes)
    if model_name == 'InceptionResNetV2':
        pre_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model,nb_classes)
    if model_name == 'DenseNet121':
        pre_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model,nb_classes)
    if model_name == 'EfficientNetB0':
        pre_model = efn.EfficientNetB0(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'EfficientNetB1':
        pre_model = efn.EfficientNetB1(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'EfficientNetB2':
        pre_model = efn.EfficientNetB2(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'EfficientNetB3':
        pre_model = efn.EfficientNetB3(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'EfficientNetB4':
        pre_model = efn.EfficientNetB4(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'EfficientNetB5':
        pre_model = efn.EfficientNetB5(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'EfficientNetB6':
        pre_model = efn.EfficientNetB6(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'EfficientNetB7':
        pre_model = efn.EfficientNetB7(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)
    if model_name == 'EfficientNetL2':
        pre_model = efn.EfficientNetL2(input_tensor=input_tensor, weights='imagenet', include_top=False)
        model = add_new_last_layer(pre_model, nb_classes)

    return model


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)

    return model


def compile_model(model, optimizer='adam',lr=1e-3, epsilon=0.001):
    if optimizer == 'sgd':
        model.compile(optimizer=SGD(lr=lr, momentum=0.9, decay=0.0001, nesterov=True),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    if optimizer == 'rmsprop':
        model.compile(optimizer=RMSprop(lr=lr, rho=0.9, epsilon=epsilon, decay=0.0),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    if optimizer == 'adagrad':
        model.compile(optimizer=Adagrad(lr=lr, epsilon=epsilon, decay=0.0),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    if optimizer == 'adadelta':
        model.compile(optimizer=Adadelta(lr=lr, rho=0.95, epsilon=epsilon, decay=0.0),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    if optimizer == 'adam':
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=0.0, amsgrad=False),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    if optimizer == 'adamax':
        model.compile(optimizer=Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=0.0),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    if optimizer == 'nadam':
        model.compile(optimizer=Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, schedule_decay=0.004),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def check_frozen(model):
    # 可训练层
    print('trainable')
    for x in model.trainable_weights:
        print(x.name)
    print('\n')
    # 不可训练层
    print('not trainable')
    for x in model.non_trainable_weights:
        print(x.name)
    print('\n')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # model_test = load_model('DenseNet121',input_shape=(1000,1000,3))
    # model_test.summary()

    # model = load_model('DenseNet121',input_shape=(1000,1000,3),nb_classes=3,pre_train=True)
    # model = compile_model(model, optimizer='adam', lr=1e-4)
    #
    # check_frozen(model)

    # plot_model(model_test, to_file='check/model_DenseNet121.png', show_shapes=True, show_layer_names=True)

    model_pretrain = load_imagenet_model('EfficientNetB1',input_shape=(1000,1000,3), pool_size=3)
    model_pretrain.summary()
    # plot_model(model_pretrain, to_file='check/model_EfficientNetB1.png', show_shapes=True, show_layer_names=True)

    # model_pretrain.save_weights('check/pre_train.h5')
    # plot_model(model_pretrain, to_file='check/model_pre_DenseNet121.png', show_shapes=True, show_layer_names=True)
    #

    # print(model_test.get_layer('conv5_block13_1_conv').get_weights())
    # print('')
    # model_test.load_weights('check/pre_train.h5',by_name=True)
    # print(model_test.get_layer('conv5_block13_1_conv').get_weights())
    # print(type(model_test.weights))
