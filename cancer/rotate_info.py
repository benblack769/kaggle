import scipy.ndimage
import numpy as np
import os
from PIL import Image
import time
import random
import sys
from multiprocessing import Process, Queue, cpu_count

#if not os.path.exists("rotate_examples/"):
#    os.mkdir("rotate_examples/")

#trainx = np.load("../data/in_data.npy")
#trainy = np.load("../data/out_data.npy")

def shuffle_together(tx,ty):
    assert len(tx) == len(ty)
    idxs = np.arange(len(tx))
    np.random.shuffle(idxs)
    new_tx = tx[idxs]
    new_ty = ty[idxs]
    del tx
    del ty
    return new_tx,new_ty

#def batch_generator()

CHUNK_SIZE = 32
BATCH_SIZE = 2048
SHUFFLE_SIZE = 1024*8

def file_chunk_generator():
    folder = "../data/chunked_train/"
    num_inp = len(list(os.listdir(folder)))//2
    while True:
        idx = random.randrange(num_inp)*CHUNK_SIZE
        inp = np.load(folder+"in{}.npy".format(idx))
        out = np.load(folder+"out{}.npy".format(idx))
        yield inp, out

def chunk_generator(inx,iny):
    while True:
        inx,iny = shuffle_together(inx,iny)
        for x in range(0,len(inx),CHUNK_SIZE):
            yield inx[x:x+CHUNK_SIZE],inx[x:x+CHUNK_SIZE]

def rotate_generator(chunk_gen):
    for chunkx,chunky in chunk_gen:
        rotatedx = scipy.ndimage.rotate(chunkx,random.random()*360,axes=(2,1),reshape=False)
        yield rotatedx,chunky

def shuffle_generator(chunk_gen):
    while True:
        all_shuffle_data = [next(chunk_gen) for _ in range(SHUFFLE_SIZE//CHUNK_SIZE)]
        sx = np.concatenate([d[0] for d in all_shuffle_data],axis=0)
        sy = np.concatenate([d[1] for d in all_shuffle_data],axis=0)

        sx,sy = shuffle_together(sx,sy)

        for x in range(0, SHUFFLE_SIZE, BATCH_SIZE):
            yield sx[x:x+BATCH_SIZE], sy[x:x+BATCH_SIZE]

def full_file_generator():
    file_gen = file_chunk_generator()
    rotate_gen = rotate_generator(file_gen)
    return shuffle_generator(rotate_gen)

def multiproc_gen_helper(queue):
    for batchx, batchy in full_file_generator():
        queue.put((batchx,batchy))

def multiproc_generator():
    queue = Queue(50)
    procs = [Process(target=multiproc_gen_helper,args=(queue,)) for p in range(4)]
    for p in procs:
        p.start()
    while True:
        yield queue.get()

def profile_gen(generator):
    start = time.time()
    for x in range(100):
        next(generator)
    end = time.time()
    print("gen time: ",end-start)

if __name__ == "__main__":
    profile_gen(multiproc_generator())
    sys.stdout.flush()
    profile_gen(full_file_generator())
    sys.stdout.flush()

#section = trainx[:1024]
#rotated = scipy.ndimage.rotate(section,70,axes=(2,1),reshape=False)


#start = time.clock()
#for x in range(1024):
#    rotated = scipy.ndimage.rotate(section[x],30,axes=(2,1),reshape=False)
#end = time.clock()
#print("time_seperate: ",end - start)

#for x in range(20):
#    img = Image.fromarray(rotated[x])
#    img.save("rotate_examples/example{}.bmp".format(x))
