import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

class Net(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(6, kernel_size=5)
            self.pool1 = nn.MaxPool2D(pool_size=2)
            self.conv2 = nn.Conv2D(16, kernel_size=5)
            self.pool2 = nn.MaxPool2D(pool_size=2)
            self.fc1 = nn.Dense(120)
            self.fc2 = nn.Dense(84)
            # You can use a Dense layer for fc3 but we do dot product manually
            # here for illustration purposes.
            self.fc3_weight = self.params.get('fc3_weight', shape=(10, 84))

    def hybrid_forward(self, F, x, fc3_weight):
        # Here `F` can be either mx.nd or mx.sym, x is the input data,
        # and fc3_weight is either self.fc3_weight.data() or
        # self.fc3_weight.var() depending on whether x is Symbol or NDArray
        print(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dot(x, fc3_weight, transpose_b=True)
        return x
net = Net()
net.initialize()
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)

net.hybridize()
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
x = mx.nd.random_normal(shape=(16, 1, 28, 28))
net(x)
