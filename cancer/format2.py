import pandas
import numpy as np
from PIL import Image


def read_img(fname):
    im = Image.open(fname)
    imarray = np.array(im)
    return imarray

def read_data(folder):
    dataframe = pandas.read_csv(folder+"train_labels.csv")
    filenames = folder + "train/" + dataframe['id'] + ".tif"
    output_data = dataframe['label']
    input_data = np.stack(read_img(fname) for fname in filenames)
    return input_data,np.asarray(list(output_data))

in_data,out_data = read_data("../data/")

np.save("../data/in_data.npy",in_data)
np.save("../data/out_data.npy",out_data)
