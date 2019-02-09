from keras.models import Sequential, Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation, Lambda, Dense, Flatten, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import pandas
from keras import backend as K
from keras.layers import Layer
import numpy as np
from PIL import Image
import numpy as np
import mxnet as mx
import os
import shutil
#mxnet specific
from keras.backend import KerasSymbol,keras_mxnet_symbol

trainx = np.load("../data/in_data.npy")
trainy = np.load("../data/out_data.npy")
#datagen = ImageDataGenerator(
        #rotation_range=40,
#        rescale=1./255)

#flow_from_dataframe(dataframe, directory, x_col='filename', y_col='class', has_ext=True, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest')
#IMAGE_WIDTH = 96
'''image_gen = datagen.flow_from_dataframe(
    dataframe,
    "../data/bmps_train/",
    x_col='fname',
    y_col='strlabel',
    target_size=(96,96),
    batch_size=16,
    class_mode='binary',
    save_format='bmp',
)'''
NUM_FILTERS = 32
IMG_SIZE = 96
IMG_CHANNELS = 3
BATCH_SIZE = 16

class GroupedConvLayer(Layer):
    def __init__(self, conv_size, num_paths, path_size,  **kwargs):
        super(GroupedConvLayer, self).__init__(**kwargs)
        self.conv_size = conv_size
        self.num_paths = num_paths
        self.path_size = path_size

        self.hidden_size = num_paths * path_size

    def build(self, input_shape):
        CONV_SIZE = [3,3]
        self.input_channels = input_shape[1]
        # Create a trainable weight variable for this layer.
        self.convweight = self.add_weight(name='convweight',
                                      shape=(self.hidden_size, self.path_size, self.conv_size[0],self.conv_size[1]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.convbias = self.add_weight(name='convbias',
                                      shape=(self.hidden_size,),
                                      initializer='zeros',
                                      trainable=True)
        super(GroupedConvLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        pad = self.conv_size[0]//2, self.conv_size[1]//2
        print(x.symbol.infer_shape())
        conved = mx.sym.Convolution(
            data=x.symbol,
            weight=self.convweight.symbol,
            bias=self.convbias.symbol,
            kernel=self.conv_size,
            num_filter=self.hidden_size,
            num_group=self.num_paths,
            pad=pad,
            layout='NCHW',
        )
        scaled = conved * self.num_paths
        print("conved")
        print(conved.infer_shape())
        #print(conved.shape)
        print(conved)
        return KerasSymbol(scaled,is_var=True)

    def compute_output_shape(self, input_shape):
        print("input_shape")
        print(input_shape)
        return input_shape

'''@keras_mxnet_symbol
def custom_op(x):
    #return K.sqrt(x)
    return KerasSymbol(mx.sym.sqrt(data=x.symbol))
'''
def transpose_input(x):
     res = K.reshape(x,(BATCH_SIZE,IMG_SIZE,IMG_SIZE,IMG_CHANNELS))
     res = K.permute_dimensions(res,[0,3,1,2])
     return res

def check_shape(x):
    print("check_shape")
    print(x.symbol.infer_shape())
    return x


def build_model():
    MAIN_SIZE = 32
    DEPTH=5
    data_format = 'channels_first'
    NUM_LAYERS_PER_DEPTH = 1
    NUM_PATHS = 16
    PATH_SIZE = 8
    model = Sequential([
        Lambda(transpose_input,input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
        #Lambda(check_shape),
        #Conv2D(32,[3,3],padding='same',data_format=data_format),#, input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
        #Lambda(check_shape),
    ])
    for x in range(DEPTH):
        for y in range(NUM_LAYERS_PER_DEPTH):
            #model.add(BatchNormalization(axis=1))
            #model.add(Conv2D(NUM_PATHS*PATH_SIZE, [1,1], data_format=data_format, use_bias=False))
            model.add(BatchNormalization(axis=1))
            model.add(Conv2D(MAIN_SIZE, [3,3], padding='same',data_format=data_format, use_bias=True, kernel_initializer='random_normal'))
            model.add(Activation('relu'))
            model.add(BatchNormalization(axis=1))
            model.add(Conv2D(MAIN_SIZE, [3,3], padding='same',data_format=data_format, use_bias=True, kernel_initializer='random_normal'))
            #model.add(GroupedConvLayer(
            #    conv_size=[3,3],
            #    num_paths=NUM_PATHS,
            #    path_size=PATH_SIZE))
            model.add(Activation('relu'))
            #model.add(Conv2D(MAIN_SIZE, [1,1], data_format=data_format, use_bias=False))
            #model.add(Conv2D(MAIN_SIZE, [3,3], data_format=data_format, use_bias=True, kernel_initializer='random_normal'))

        model.add(MaxPooling2D(pool_size=(2,2),padding='same',data_format=data_format))
        #model.add(Dense(MAIN_SIZE))

    #model.add(Flatten())
    #model.add(Dense(MAIN_SIZE,activation='relu'))
    #model.add(Dense(1))
    #model.add(Lambda(lambda x: x * 0.2))
    model.add(Conv2D(1, [1,1], data_format=data_format, use_bias=True))
    model.add(GlobalAveragePooling2D(data_format=data_format))
    model.add(Activation('sigmoid'))

    return model


model = build_model()
'''Sequential([
    Lambda(transpose_input,input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
    Conv2D(32,[3,3],padding='same'),#, input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
    Activation('relu'),
    Dense(32),
    GroupedConvLayer(conv_size=[3,3], num_paths=4, path_size=8),
    Dense(32),
    Conv2D(32,[3,3],padding='same'),
    #Activation('relu'),
    #Lambda(custom_op),
    #Activation('relu'),
    Conv2D(1,[3,3],padding='same'),
    GlobalAveragePooling2D(),
    Lambda(lambda x: x * 0.02),
    Activation('sigmoid'),
])'''
print("model finished!")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

def shuffle_together(tx,ty):
    assert len(tx) == len(ty)
    idxs = np.arange(len(tx))
    np.random.shuffle(idxs)
    new_tx = tx[idxs]
    new_ty = ty[idxs]
    del tx
    del ty
    return new_tx,new_ty

def round_div(x,div):
    return x - x % div

trainx, trainy = shuffle_together(trainx, trainy)

validation_amount = 1024*32
cutoffbatch_size = round_div(len(trainx)-validation_amount, BATCH_SIZE)
#cutoffbatch_size = 4096
testx =  trainx[-validation_amount:]
testy =  trainy[-validation_amount:]
trainx = trainx[:cutoffbatch_size]
trainy = trainy[:cutoffbatch_size]

weights = "../data/weights/"
if os.path.exists(weights):
    shutil.rmtree(weights)
os.mkdir(weights)

for x in range(10):
    trainx, trainy = shuffle_together(trainx, trainy)

    model.fit(
        x=trainx,
        y=trainy,
        batch_size=BATCH_SIZE,
        epochs=2,
        shuffle=True,
        validation_data=(testx, testy)
    )
    model.save_weights("../data/weights/step{}.h5".format(x))
    #exit(0)
#model.fit_generator(
#    image_gen,
#    steps_per_epoch=16,
#    epochs=50
#)
