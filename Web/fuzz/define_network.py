import caffe
from caffe import layers as L
from caffe import params as P


def lenet(lmdb, batch_size):
    n = caffe.NetSpec()
    #  输入层
    n.data, n.label = L.Data(batch_size=batch_size,
                             backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)
    #  卷积层
    n.conv1 = L.Convolution(n.data, kernel_size=5,
                            num_output=20, weight_filler=dict(type='xavier'))
    #  池化层
    n.pool1 = L.Pooling(n.conv1, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    #  卷积层
    n.conv2 = L.Convolution(n.pool1, kernel_size=5,
                            num_output=50, weight_filler=dict(type='xavier'))
    #  池化层
    n.pool2 = L.Pooling(n.conv2, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    #  全连接层
    n.ip1 = L.InnerProduct(n.pool2, num_output=500,
                           weight_filler=dict(type='xavier'))
    #  激活函数
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    #  全连接层
    n.ip2 = L.InnerProduct(n.relu1, num_output=10,
                           weight_filler=dict(type='xavier'))
    #  损失函数
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()


with open('mnist/lenent_auto_train.prototxt', 'w')as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w')as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))
