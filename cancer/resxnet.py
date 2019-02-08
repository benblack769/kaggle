import tensorflow as tf

'''def gather_fn(maps,convsize):
    # only odd values are symetric, only semetric gathers are supported
    assert convsize[0] % 2 == 1 and convsize[1] % 2 == 1
    size = maps.get_shape().as_list()
    convsizehalf = [convsize[0]//2,convsize[1]//2]
    padding = [[0,0],[convsizehalf[0],convsizehalf[0]],[convsizehalf[1],convsizehalf[1]]]
    all_padded = tf.pad(maps,padding)
    all_cropped = []
    for x in range(convsize[0]):
        for y in range(convsize[1]):
            cropped = tf.slice(all_padded,(0,x,y),(size[0],size[1],size[2]))
            reshaped = tf.reshape(cropped,(size[0],size[1],size[2],1))
            all_cropped.append(reshaped)
    return tf.concat(all_cropped,axis=3)
'''

class ResXNet:
    def __init__(self,input_channels,filters,padding,kernel_size,num_branches,inner_size,activation=None,strides=[1,1,1,1]):
        self.input_channels = input_channels
        self.output_channels = filters
        self.num_branches = num_branches
        self.inner_size = inner_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

        self.hidden_size = self.inner_size * self.num_branches
        self.in_to_hidden = tf.layers.Dense(
            units=self.hidden_size,
            activation=activation,
        )
        self.total_kern_size = kernel_size[0] * kernel_size[1]
        #self.conv_manipulator = tf.Variable(make_gather_conv(kernel_size[0],kernel_size[1],1),dtype=tf.float32,trainable=False)
        self.all_convs = [tf.layers.Conv2D(
            filters=self.inner_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=tf.nn.relu) for _ in range(self.num_branches)]

        self.hidden_to_out = tf.layers.Dense(
            units=self.output_channels,
            activation=None,
        )
        #self.conv_evaluator = tf.Variable(np.random.normal(size=(self.hidden_size,self.total_kern_size)),dtype=tf.float32)
        pass

    def apply(self,tensor4d):
        conv1x1out = self.in_to_hidden(tensor4d)
        in_shape = conv1x1out.get_shape().as_list()
        reshaped_hidden = tf.reshape(conv1x1out,[in_shape[0],in_shape[1],in_shape[2],self.num_branches,self.inner_size])
        #transposed_hidden = tf.t
        paths = []
        for x in range(self.num_branches):
            branch = tf.slice(reshaped_hidden,[0,0,0,x,0],[in_shape[0],in_shape[1],in_shape[2],1,self.inner_size])
            reshape_brach = tf.reshape(branch,[in_shape[0],in_shape[1],in_shape[2],self.inner_size])
            conved = self.all_convs[x](reshape_brach)
            paths.append(conved)

        all_paths = tf.concat(paths,axis=3)

        final_out = self.hidden_to_out(all_paths)
        return final_out


    __call__=apply
