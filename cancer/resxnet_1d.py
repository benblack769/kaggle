import tensorflow as tf

class ResXNet1d:
    def __init__(self,input_channels,filters,num_folds,activation=None):
        self.input_channels = input_channels
        self.output_channels = filters
        self.num_folds = num_folds
        self.activation = activation

        self.hidden_size = self.num_folds * self.output_channels
        self.dense_layer = tf.layers.Dense(
            units=self.hidden_size,
            activation=activation,
        )

    def apply_no_sum(self,tensor4d):
        conv1x1out = self.dense_layer(tensor4d)

        return conv1x1out

    def apply(self,tensor4d):
        out_val = self.apply_no_sum(tensor4d)
        in_shape = out_val.get_shape().as_list()
        reshaped_for_sum = tf.reshape(out_val,(in_shape[0],in_shape[1],in_shape[2],in_shape[3]//self.num_folds,self.num_folds))
        final_out = tf.reduce_mean(reshaped_for_sum,axis=4)
        return final_out


    __call__=apply
