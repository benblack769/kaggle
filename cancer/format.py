import os
from PIL import Image

def reformat_all(folder,outfolder):
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    for fname in os.listdir(folder):
        save_img(folder+fname,outfolder+fname+".bmp")

def save_img(infname,outfname):
    im = Image.open(infname)
    im.save(outfname)
    #imarray = np.array(im)
    #print(imarray.shape)

def read_img(fname):
    im = Image.open(fname)
    imarray = np.array(im)
    print(imarray.shape)

reformat_all("../data/test/","../data/bmps_test/")
