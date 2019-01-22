import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 16
NUM_TRAIN_ITERS = 1000
IMG_SIZE = 96
IMG_CHANNELS = 3
SUPERPIX_CHANNELS = 32

def read_csv(filename):
    with open(filename) as file:
        first_line = file.readline()
        headers = first_line.strip().split(',')
        csv = {head:[] for head in headers}
        for data_line in file.readlines():
            datas = data_line.strip().split(',')
            for idx,data in enumerate(datas):
                csv[headers[idx]].append(data)
        return csv

def shape_image(img):
    return tf.to_float(tf.reshape(img,[IMG_SIZE,IMG_SIZE,IMG_CHANNELS])) / 255.0

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_bmp(image_string)
    image_resized = shape_image(image_decoded)
    return image_resized, label

def make_dataset(csv_data,folder):
    filenames = [folder+id+".tif.bmp" for id in csv_data["id"]]
    filenames_tensor = tf.constant(filenames)
    #print(filenames_tensor)
    outputs_tensor = tf.constant([float(l) for l in csv_data['label']],dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((filenames_tensor, outputs_tensor))
    ds = ds.shuffle(100000)
    ds = ds.map(_parse_function)
    ds = ds.repeat(count=10000000000000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda in1,in2: (tf.reshape(in1,[BATCH_SIZE,IMG_SIZE,IMG_SIZE,IMG_CHANNELS]),in2))
    ds = ds.prefetch(8)

    iter = ds.make_one_shot_iterator()
    input,output = iter.get_next()
    return input,output

class SuperPixel:
    def __init__(self):
        self.layers = {}

        self.layers["fc1"] = tf.layers.Dense(
            units=64,
            activation=tf.nn.relu
        )
        self.layers["fc2"] = tf.layers.Dense(
            units=SUPERPIX_CHANNELS,
            #activation=tf.nn.relu
        )


    def fn(self,input):
        lay1 = self.layers['fc1'](input)
        laycmp = self.layers['fc2'](input)
        return laycmp

class Model:
    def __init__(self):
        self.layers = {}
        CONV1_SIZE=[3,3]
        POOL_SIZE=[2,2]
        POOL_STRIDES=[2,2]
        lay1size = 32
        self.DEPTH=DEPTH=5

        self.layers["input_filter"] = tf.layers.Dense(
            units=lay1size,
            activation=tf.nn.relu
        )

        for x in range(DEPTH):
            self.layers["convlay{}0".format(x)] = tf.layers.Conv2D(
                filters=lay1size,
                kernel_size=CONV1_SIZE,
                padding="same",
                activation=tf.nn.relu)
            self.layers["convlay{}1".format(x)] = tf.layers.Conv2D(
                filters=lay1size,
                kernel_size=CONV1_SIZE,
                padding="same",
                activation=tf.nn.relu)
            self.layers["maxpool{}".format(x)] = tf.layers.MaxPooling2D(
                pool_size=POOL_SIZE,
                strides=POOL_STRIDES,
                padding='same',
            )

        self.layers["fc1"] = tf.layers.Dense(
            units=lay1size,
            activation=tf.nn.relu
        )
        self.layers["fc2"] = tf.layers.Dense(
            units=lay1size,
            activation=tf.nn.relu
        )
        self.layers["fcout"] = tf.layers.Dense(
            units=1
        )

    def fn(self,input):
        cur_input = input
        cur_input = self.layers["input_filter"](cur_input)
        for x in range(self.DEPTH):
            cur_input = self.layers["convlay{}0".format(x)](cur_input)
            cur_input = self.layers["convlay{}1".format(x)](cur_input)
            cur_input = self.layers["maxpool{}".format(x)](cur_input)
            print(cur_input)

        cur_input = tf.layers.flatten(cur_input)
        cur_input = self.layers["fc1"](cur_input)
        cur_input = self.layers["fc2"](cur_input)
        cur_input = self.layers["fcout"](cur_input)
        print(cur_input)

        return cur_input

def model(input):
    mod = Model()
    cur_out = mod.fn(input)
    cur_out2 = mod.fn(0.1*input)
    #cur_out = input
    print(cur_out.shape)

    return tf.squeeze(cur_out)


def cmp_super_out(out1,out2):
    diff = out1 * out2
    #flatdiff = tf.layers.flatten(diff)
    summeddiff = tf.reduce_mean(diff,axis=1)
    return summeddiff

def compare_loss(input,superpix):
    model_out = superpix.fn(input)
    roll_batch_size = BATCH_SIZE*IMG_SIZE*IMG_SIZE
    reshaped = tf.reshape(model_out,[roll_batch_size,SUPERPIX_CHANNELS])
    roll_idx = tf.random_uniform((1,),minval=0,maxval=roll_batch_size,dtype=tf.int32)
    roll_idx = tf.reshape(roll_idx,[])
    rolled = tf.concat([reshaped[roll_idx:],reshaped[:roll_idx]],axis=0)
    return cmp_super_out(reshaped,rolled)

def train_main():
    csv_data = read_csv("train_labels.csv")
    bmp_folder = "../data/bmps_train/"
    input_gen,outputs_truth = make_dataset(csv_data,bmp_folder)

    superpix = SuperPixel()
    pix_loss = tf.reduce_mean(compare_loss(input_gen,superpix))

    model_out = model((input_gen))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs_truth,logits=model_out))

    optimizer_main = tf.train.AdamOptimizer(0.001)
    optim_main = optimizer_main.minimize(loss)

    optimizer_pix = tf.train.AdamOptimizer(0.001)
    optim_pix = optimizer_pix.minimize(pix_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for x in range(100):
            tot_loss = 0
            num_iters = 10
            #for y in range(num_iters):
            #    opt_val,loss_val = sess.run([optim_pix,pix_loss])
            #    tot_loss += loss_val
            #print(tot_loss / num_iters)

        print("main loss started")
        for x in range(NUM_TRAIN_ITERS):
            tot_loss = 0
            num_iters = 80
            for y in range(num_iters):
                opt_val,loss_val = sess.run([optim_main,loss])
                tot_loss += loss_val
            print(tot_loss / num_iters)



#read_img("examples/000a2a35668f04edebc0b06d5d133ad90c9IMG_CHANNELSa044.tif")
#make_dataset(read_csv("train_labels.csv"))
train_main()
