import tensorflow as tf
import numpy as np

def make_gather_conv(conv_size,input_size):
    res = []
    conv_size_sqr = conv_size*conv_size
    for x in range(conv_size):
        for y in range(conv_size):
            for i in range(input_size):
                for j in range(input_size*conv_size_sqr):
                    res.append(int(j // input_size == x * conv_size + y))
    res_np = np.asarray(res,dtype=np.float32)
    res_reshaped = np.reshape(res_np,(conv_size,conv_size,input_size,conv_size*conv_size*input_size))
    return res_reshaped

class ResXNet:
    def __init__(self,batch_size,input_channels,convsize,output_channels,num_folds):
        self.num_folds = num_folds
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.convsize = convsize

        self.hidden_size = self.num_folds * self.output_channels
        self.dense_layer = tf.layers.Dense(
            units=hidden_size,
            activation=None,
        )
        self.conv_manipulator = tf.Variable(make_gather_conv(convsize,1),dtype=tf.float32)

        self.conv_evaluator = tf.Variable(np.random.normal(size=(self.hidden_size,convsize*convsize,),dtype=tf.float32)
        pass

    def apply(self,tensor4d):
        conv1x1out = self.dense_layer(tensor4d)
        conv1x1out = tf.nn.relu(conv1x1out)
        convsqr_size = self.convsize * self.convsize
        in_shape = conv1x1out.get_shape.as_list()

        #flatten_img = tf.reshape(conv1x1out,(in_shape[0],in_shape[1]*in_shape[2],in_shape[3]))

        images_last_dim = tf.transpose(conv1x1out, perm=[0, 3, 1, 2])

        convable = tf.reshape(images_last_dim,(in_shape[0]*in_shape[3],in_shape[1],in_shape[2],1))

        convmaniped = tf.nn.conv2d(
            input=convable,
            filter=self.conv_manipulator,
            strides=(1,1),
            padding='same',
        )
        reshaped_for_eval = tf.reshape(convmaniped,(in_shape[0],in_shape[3],in_shape[1],in_shape[2],convsqr_size))

        conv_eval_reshape = tf.reshape(self.conv_evaluator,(1, self.hidden_size, 1, 1, convsqr_size))

        evaled = tf.reduce_mean(conv_eval_reshape*reshaped_for_trans,axis=4)
        evaled = tf.nn.relu(evaled)

        channel_last_dim = tf.transpose(evaled, perm=[0, 2, 3, 1])

        reshaped_for_sum = tf.reshape(channel_last_dim,(in_shape[0],in_shape[1],in_shape[2],in_shape[3]//self.num_folds,self.num_folds))
        final_out = tf.reduce_mean(reshaped_for_sum,axis=4)
        return final_out
        
