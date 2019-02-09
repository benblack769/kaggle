import pandas
import numpy as np
from PIL import Image
import os

CHUNK_SIZE = 32

def read_img(fname):
    im = Image.open(fname)
    imarray = np.array(im)
    return imarray

def read_data(folder):
    dataframe = pandas.read_csv(folder+"train_labels.csv")
    filenames = folder + "train/" + dataframe['id'] + ".tif"
    output_data = dataframe['label']
    input_data = [read_img(fname) for fname in filenames]
    return input_data,np.asarray(list(output_data))

os.mkdir("../data/chunked_train/")

in_data,out_data = read_data("../data/")

VALIDATION_SIZE = 1024*32

for x in range(0,len(in_data) - VALIDATION_SIZE,CHUNK_SIZE):
    np.save("../data/chunked_train/in{}.npy".format(x),np.stack(in_data[x:x+CHUNK_SIZE]))
    np.save("../data/chunked_train/out{}.npy".format(x),out_data[x:x+CHUNK_SIZE])

np.save("../data/validation_in.npy",in_data[-VALIDATION_SIZE:])
np.save("../data/validation_out.npy",out_data[-VALIDATION_SIZE:])
