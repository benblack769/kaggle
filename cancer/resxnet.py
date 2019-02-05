import tensorflow as tf
import numpy as np

def make_gather_conv(convsize1,convsize2,input_size):
    res = []
    conv_size_sqr = convsize1*convsize2
    for x in range(convsize1):
        for y in range(convsize2):
            for i in range(input_size):
                for j in range(input_size*conv_size_sqr):
                    res.append(int(j // input_size == x * convsize2 + y))
    res_np = np.asarray(res,dtype=np.float32)
    res_reshaped = np.reshape(res_np,(convsize1,convsize2,input_size,conv_size_sqr*input_size))
    return res_reshaped

class ResXNet:
    def __init__(self,input_channels,filters,padding,kernel_size,num_folds,activation=None,strides=[1,1,1,1]):
        self.input_channels = input_channels
        self.output_channels = filters
        self.kernel_size = kernel_size
        self.num_folds = num_folds
        self.strides = strides
        self.padding = padding.upper()
        self.activation = activation

        self.hidden_size = self.num_folds * self.output_channels
        self.dense_layer = tf.layers.Dense(
            units=self.hidden_size,
            activation=activation,
        )
        self.total_kern_size = kernel_size[0] * kernel_size[1]
        self.conv_manipulator = tf.Variable(make_gather_conv(kernel_size[0],kernel_size[1],1),dtype=tf.float32)

        self.conv_evaluator = tf.Variable(np.random.normal(size=(self.hidden_size,self.total_kern_size)),dtype=tf.float32)
        pass

    def apply_no_sum(self,tensor4d):
        conv1x1out = self.dense_layer(tensor4d)
        convsqr_size = self.total_kern_size
        in_shape = conv1x1out.get_shape().as_list()

        #flatten_img = tf.reshape(conv1x1out,(in_shape[0],in_shape[1]*in_shape[2],in_shape[3]))

        images_last_dim = tf.transpose(conv1x1out, perm=[0, 3, 1, 2])

        convable = tf.reshape(images_last_dim,(in_shape[0]*in_shape[3],in_shape[1],in_shape[2],1))

        convmaniped = tf.nn.conv2d(
            input=convable,
            filter=self.conv_manipulator,
            strides=self.strides,
            padding=self.padding,
        )
        reshaped_for_eval = tf.reshape(convmaniped,(in_shape[0],in_shape[3],in_shape[1],in_shape[2],convsqr_size))

        conv_eval_reshape = tf.reshape(self.conv_evaluator,(1, self.hidden_size, 1, 1, convsqr_size))

        mulled = tf.multiply(conv_eval_reshape,reshaped_for_eval,name="mulfinallarge")
        evaled = tf.reduce_mean(mulleds,axis=4)
        #evaled = tf.nn.relu(evaled)

        channel_last_dim = tf.transpose(evaled, perm=[0, 2, 3, 1])
        return channel_last_dim

    def apply(self,tensor4d):
        out_val = self.apply_no_sum(tensor4d)
        in_shape = out_val.get_shape().as_list()
        reshaped_for_sum = tf.reshape(out_val,(in_shape[0],in_shape[1],in_shape[2],in_shape[3]//self.num_folds,self.num_folds))
        final_out = tf.reduce_mean(reshaped_for_sum,axis=4)
        return final_out


    __call__=apply
