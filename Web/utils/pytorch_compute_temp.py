import torch
import torch.nn.functional as F


def pytorch_compute_single(im, target_interface, GPU_mode=1):
    input_pytorch_value = None
    output_pytorch_value = None
    input_pytorch_cpu_value = None
    output_pytorch_cpu_value = None
    pytorch_shape = [1]
    for shape_element in im.shape:
        pytorch_shape.append(shape_element)
    input_pytorch = torch.reshape(torch.from_numpy(im), pytorch_shape)
    if GPU_mode != 0:
        input_pytorch_cpu_value = input_pytorch.numpy()

        if target_interface == 'conv1':
            weights_torch = torch.from_numpy(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
            output_pytorch_cpu = torch.from_numpy(
                F.conv2d(torch.from_numpy(input_pytorch.numpy().transpose((0, 3, 1, 2))), weights_torch, padding=0,
                         stride=4).numpy().transpose((0, 2, 3, 1)))
        elif target_interface == 'conv2':
            x_torch = torch.from_numpy(input_pytorch.numpy().transpose((0, 3, 1, 2)).astype(np.float64))
            weights_torch = torch.from_numpy(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
            stride = 4
            if x_torch.numpy().shape[2] % stride == 0:
                pad = max(weights_torch.numpy().shape[2] - stride, 0)
            else:
                pad = max(weights_torch.numpy().shape[2] - (x_torch.numpy().shape[2] % stride), 0)

            if pad % 2 == 0:
                pad_val = pad // 2
                padding = (pad_val, pad_val, pad_val, pad_val)
            else:
                pad_val_start = pad // 2
                pad_val_end = pad - pad_val_start
                padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
            x_torch = F.pad(x_torch, padding, "constant", 0)
            output_pytorch_cpu = torch.from_numpy(
                F.conv2d(x_torch, weights_torch, padding=0, stride=stride).numpy().transpose((0, 2, 3, 1)))
        elif target_interface == 'pool1':
            output_pytorch_cpu = torch.from_numpy(np.rollaxis(
                F.max_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_cpu_value, 3, 1)), kernel_size=(2, 2),
                             stride=(2, 2)).numpy(), 1, 4))
        elif target_interface == 'pool2':
            output_pytorch_cpu = torch.from_numpy(np.rollaxis(
                F.avg_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_cpu_value, 3, 1)), kernel_size=(2, 2),
                             stride=(2, 2)).numpy(), 1, 4))
        elif target_interface == 'relu1':
            output_pytorch_cpu = F.relu(input_pytorch)
        elif target_interface == 'dense1':
            output_pytorch_cpu = None
        elif target_interface == 'sigmoid1':
            output_pytorch_cpu = F.sigmoid(input_pytorch)
        elif target_interface == 'tanh1':
            output_pytorch_cpu = F.tanh(input_pytorch)
        else:
            output_pytorch_cpu = None
        output_pytorch_cpu_value = output_pytorch_cpu.numpy()
    if GPU_mode != 2:
        input_pytorch_value = input_pytorch.to("cuda").to("cpu").numpy()
        if target_interface == 'conv1':
            weights_torch = torch.from_numpy(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
            output_pytorch = torch.from_numpy(
                F.conv2d(torch.from_numpy(input_pytorch.numpy().transpose((0, 3, 1, 2))).to("cuda"),
                         weights_torch.to("cuda"), padding=0, stride=4).to("cpu").numpy().transpose((0, 2, 3, 1))).to(
                "cuda")
        elif target_interface == 'conv2':
            x_torch = torch.from_numpy(input_pytorch.numpy().transpose((0, 3, 1, 2)).astype(np.float64))
            weights_torch = torch.from_numpy(np.full((11, 11, 3, 1), 1, np.float64).transpose((3, 2, 0, 1)))
            stride = 4
            if x_torch.numpy().shape[2] % stride == 0:
                pad = max(weights_torch.numpy().shape[2] - stride, 0)
            else:
                pad = max(weights_torch.numpy().shape[2] - (x_torch.numpy().shape[2] % stride), 0)

            if pad % 2 == 0:
                pad_val = pad // 2
                padding = (pad_val, pad_val, pad_val, pad_val)
            else:
                pad_val_start = pad // 2
                pad_val_end = pad - pad_val_start
                padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
            x_torch = F.pad(x_torch, padding, "constant", 0)
            output_pytorch = torch.from_numpy(
                F.conv2d(x_torch.to("cuda"), weights_torch.to("cuda"), padding=0, stride=4).to("cpu").numpy().transpose(
                    (0, 2, 3, 1))).to("cuda")
        elif target_interface == 'pool1':
            output_pytorch = torch.from_numpy(np.rollaxis(
                F.max_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_value, 3, 1)).to("cuda"), kernel_size=(2, 2),
                             stride=(2, 2)).to("cpu").numpy(), 1, 4)).to("cuda")
        elif target_interface == 'pool2':
            output_pytorch = torch.from_numpy(np.rollaxis(
                F.avg_pool2d(torch.from_numpy(np.rollaxis(input_pytorch_value, 3, 1)).to("cuda"), kernel_size=(2, 2),
                             stride=(2, 2)).to("cpu").numpy(), 1, 4)).to("cuda")
        elif target_interface == 'relu1':
            output_pytorch = F.relu(input_pytorch.to("cuda"))
        elif target_interface == 'dense1':
            output_pytorch = None
        elif target_interface == 'sigmoid1':
            output_pytorch = F.sigmoid(input_pytorch.to("cuda"))
        elif target_interface == 'tanh1':
            output_pytorch = F.tanh(input_pytorch.to("cuda"))
        else:
            output_pytorch = None
        output_pytorch_value = output_pytorch.to("cpu").numpy()
    return output_pytorch_value, input_pytorch_value, output_pytorch_cpu_value, input_pytorch_cpu_value


# coding: utf-8
import numpy as np
import tensorflow as tf


class TopologyElement(object):
    def __init__(self, name, related_interface, data, op):
        """
        用于存储算子组合拓扑结构信息
        :param name:结构名称,str
        :param related_interface:相关的单一算子，List(str)
        :param data 输入占位
        :param op 运算过程，由于Tensorflow的特性，运算过程可以使用变量代替，但是要传递额外参数输入
        """
        self.name = name
        self.related_interface = related_interface
        self.data = data
        self.op = op

    def match(self, target_interface):
        idx = 0
        for idx in range(0, len(target_interface)):
            if str.isdigit(target_interface[idx]):
                break
        pattern = target_interface[0:idx]
        if pattern in self.related_interface:
            return True
        return False

    def getName(self):
        return self.name

    def getOutput(self):
        return self.related_interface[-1]

    def getData(self):
        return self.data

    def setOp(self, op):
        self.op = op

    def getOp(self):
        return self.op


def generate_topologies():
    data_holder = tf.placeholder(tf.float64)
    # Lenet结构
    # conv_pool1
    conv_pool1 = TopologyElement('conv_pool1', ['conv', 'pool'], data_holder, None)
    conv_filter = tf.get_variable('weights_cpu1', [5, 5, 3, 20],
                                  initializer=tf.constant_initializer(1),
                                  dtype=tf.float64)
    conv_pool1_conv1 = tf.nn.convolution(data_holder, filter=conv_filter, strides=[1, 1], padding='VALID')
    conv_pool1_pool1 = tf.nn.max_pool(conv_pool1_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv_pool1.setOp(conv_pool1_pool1)

    # conv_pool2
    conv_pool2 = TopologyElement('conv_pool2', ['conv', 'pool'], data_holder, None)
    conv_filter = tf.get_variable('weights_cpu2', [5, 5, 3, 50],
                                  initializer=tf.constant_initializer(1),
                                  dtype=tf.float64)
    conv_pool2_conv1 = tf.nn.convolution(data_holder, filter=conv_filter, strides=[1, 1], padding='VALID')
    conv_pool2_pool1 = tf.nn.max_pool(conv_pool2_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv_pool2.setOp(conv_pool2_pool1)

    # relu_softmax1
    relu_softmax1 = TopologyElement('relu_softmax1', ['relu', 'softmax'], data_holder, None)
    relu_softmax1_relu1 = tf.nn.relu(data_holder)
    relu_softmax1_softmax1 = tf.nn.softmax(relu_softmax1_relu1)
    relu_softmax1.setOp(relu_softmax1_softmax1)

    # Vgg结构
    # conv_relu_pool1
    conv_relu_pool1 = TopologyElement('conv_relu_pool1', ['conv', 'relu', 'pool'], data_holder, None)
    conv_filter_1 = tf.get_variable('weights_conv_relu_pool1_1', [3, 3, 3, 64],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool1_conv1 = tf.nn.convolution(data_holder, filter=conv_filter_1, strides=[1, 1], padding='VALID')
    conv_relu_pool1_relu1 = tf.nn.relu(conv_relu_pool1_conv1)
    conv_filter_2 = tf.get_variable('weights_conv_relu_pool1_2', [3, 3, 3, 64],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool1_conv2 = tf.nn.convolution(conv_relu_pool1_relu1, filter=conv_filter_2, strides=[1, 1],
                                              padding='VALID')
    conv_relu_pool1_relu2 = tf.nn.relu(conv_relu_pool1_conv2)
    conv_relu_pool1_pool1 = tf.nn.max_pool(conv_relu_pool1_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                           padding='VALID')
    conv_relu_pool1.setOp(conv_relu_pool1_pool1)

    # conv_relu_pool2
    conv_relu_pool2 = TopologyElement('conv_relu_pool2', ['conv', 'relu', 'pool'], data_holder, None)
    conv_filter_1 = tf.get_variable('weights_conv_relu_pool2_1', [3, 3, 3, 128],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool2_conv1 = tf.nn.convolution(data_holder, filter=conv_filter_1, strides=[1, 1], padding='VALID')
    conv_relu_pool2_relu1 = tf.nn.relu(conv_relu_pool2_conv1)
    conv_filter_2 = tf.get_variable('weights_conv_relu_pool2_2', [3, 3, 3, 128],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool2_conv2 = tf.nn.convolution(conv_relu_pool2_relu1, filter=conv_filter_2, strides=[1, 1],
                                              padding='VALID')
    conv_relu_pool2_relu2 = tf.nn.relu(conv_relu_pool2_conv2)
    conv_relu_pool2_pool1 = tf.nn.max_pool(conv_relu_pool2_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                           padding='VALID')
    conv_relu_pool2.setOp(conv_relu_pool2_pool1)

    # conv_relu_pool3
    conv_relu_pool3 = TopologyElement('conv_relu_pool3', ['conv', 'relu', 'pool'], data_holder, None)
    conv_filter_1 = tf.get_variable('weights_conv_relu_pool3_1', [3, 3, 3, 256],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool3_conv1 = tf.nn.convolution(data_holder, filter=conv_filter_1, strides=[1, 1], padding='VALID')
    conv_relu_pool3_relu1 = tf.nn.relu(conv_relu_pool3_conv1)
    conv_filter_2 = tf.get_variable('weights_conv_relu_pool3_2', [3, 3, 3, 256],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool3_conv2 = tf.nn.convolution(conv_relu_pool3_relu1, filter=conv_filter_2, strides=[1, 1],
                                              padding='VALID')
    conv_relu_pool3_relu2 = tf.nn.relu(conv_relu_pool3_conv2)
    conv_filter_3 = tf.get_variable('weights_conv_relu_pool3_2', [3, 3, 3, 256],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool3_conv3 = tf.nn.convolution(conv_relu_pool3_relu2, filter=conv_filter_3, strides=[1, 1],
                                              padding='VALID')
    conv_relu_pool3_relu3 = tf.nn.relu(conv_relu_pool3_conv3)
    conv_relu_pool3_pool1 = tf.nn.max_pool(conv_relu_pool3_relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                           padding='VALID')
    conv_relu_pool3.setOp(conv_relu_pool3_pool1)

    # conv_relu_pool4
    conv_relu_pool4 = TopologyElement('conv_relu_pool4', ['conv', 'relu', 'pool'], data_holder, None)
    conv_filter_1 = tf.get_variable('weights_conv_relu_pool4_1', [3, 3, 3, 256],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool4_conv1 = tf.nn.convolution(data_holder, filter=conv_filter_1, strides=[1, 1], padding='VALID')
    conv_relu_pool4_relu1 = tf.nn.relu(conv_relu_pool4_conv1)
    conv_filter_2 = tf.get_variable('weights_conv_relu_pool4_2', [3, 3, 3, 256],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool4_conv2 = tf.nn.convolution(conv_relu_pool4_relu1, filter=conv_filter_2, strides=[1, 1],
                                              padding='VALID')
    conv_relu_pool4_relu2 = tf.nn.relu(conv_relu_pool4_conv2)
    conv_filter_3 = tf.get_variable('weights_conv_relu_pool4_2', [3, 3, 3, 256],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool4_conv3 = tf.nn.convolution(conv_relu_pool4_relu2, filter=conv_filter_3, strides=[1, 1],
                                              padding='VALID')
    conv_relu_pool4_relu3 = tf.nn.relu(conv_relu_pool4_conv3)
    conv_filter_4 = tf.get_variable('weights_conv_relu_pool4_2', [3, 3, 3, 256],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_relu_pool4_conv4 = tf.nn.convolution(conv_relu_pool4_relu3, filter=conv_filter_4, strides=[1, 1],
                                              padding='VALID')
    conv_relu_pool4_relu4 = tf.nn.relu(conv_relu_pool4_conv4)
    conv_relu_pool4_pool1 = tf.nn.max_pool(conv_relu_pool4_relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                           padding='VALID')
    conv_relu_pool4.setOp(conv_relu_pool4_pool1)

    # SE-BN-Inception
    # conv_sigmoid1
    conv_sigmoid1 = TopologyElement('conv_sigmoid1', ['conv', 'sigmoid'], data_holder, None)
    conv_filter_1 = tf.get_variable('weights_conv_sigmoid1', [1, 1, 3, 320],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_sigmoid1_conv1 = tf.nn.convolution(data_holder, filter=conv_filter_1, strides=[1, 1],
                                            padding='VALID')
    conv_sigmoid1_sigmoid1 = tf.nn.sigmoid(conv_sigmoid1_conv1)
    conv_sigmoid1.setOp(conv_sigmoid1_sigmoid1)

    # SE-ResNet-50
    # conv_sigmoid2
    conv_sigmoid2 = TopologyElement('conv_sigmoid2', ['conv', 'sigmoid'], data_holder, None)
    conv_filter_1 = tf.get_variable('weights_conv_sigmoid2', [1, 1, 3, 320],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_sigmoid2_conv1 = tf.nn.convolution(data_holder, filter=conv_filter_1, strides=[1, 1],
                                            padding='VALID')
    conv_sigmoid2_sigmoid1 = tf.nn.sigmoid(conv_sigmoid2_conv1)
    conv_sigmoid2.setOp(conv_sigmoid2_sigmoid1)

    # conv_sigmoid3
    conv_sigmoid3 = TopologyElement('conv_sigmoid3', ['conv', 'sigmoid'], data_holder, None)
    conv_filter_1 = tf.get_variable('weights_conv_sigmoid3', [1, 1, 3, 512],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_sigmoid3_conv1 = tf.nn.convolution(data_holder, filter=conv_filter_1, strides=[1, 1],
                                            padding='VALID')
    conv_sigmoid3_sigmoid1 = tf.nn.sigmoid(conv_sigmoid3_conv1)
    conv_sigmoid3.setOp(conv_sigmoid3_sigmoid1)

    # conv_sigmoid4
    conv_sigmoid4 = TopologyElement('conv_sigmoid4', ['conv', 'sigmoid'], data_holder, None)
    conv_filter_1 = tf.get_variable('weights_conv_sigmoid4', [1, 1, 3, 1024],
                                    initializer=tf.constant_initializer(1),
                                    dtype=tf.float64)
    conv_sigmoid4_conv1 = tf.nn.convolution(data_holder, filter=conv_filter_1, strides=[1, 1],
                                            padding='VALID')
    conv_sigmoid4_sigmoid1 = tf.nn.sigmoid(conv_sigmoid4_conv1)
    conv_sigmoid4.setOp(conv_sigmoid4_sigmoid1)

    return (
        conv_pool1,
        conv_pool2,
        relu_softmax1,
        conv_relu_pool1,
        conv_relu_pool2,
        conv_relu_pool3,
        conv_relu_pool4,
        conv_sigmoid1,
        conv_sigmoid2,
        conv_sigmoid3,
        conv_sigmoid4
    )


topologies = generate_topologies()


def tensorflow_compute_single(data, target_interface, GPU_mode=1):
    input_tensorflow_gpu_value = None
    output_tensorflow_gpu_value = None
    input_tensorflow_cpu_value = None
    output_tensorflow_cpu_value = None

    # 获取Tensor规模
    tensorflow_shape = [1]
    for shape_element in data.shape:
        tensorflow_shape.append(shape_element)
    # input_tensorflow = tf.reshape(np.array(data).astype(np.float32), tensorflow_shape)
    input_tensorflow = tf.reshape(data, tensorflow_shape)

    # 确定tensorflow接口参数
    if target_interface == 'conv1':
        conv_filter = tf.get_variable('weights_cpu', [11, 11, 3, 1],
                                      initializer=tf.constant_initializer(1),
                                      dtype=tf.float64)
        output_tensorflow = tf.nn.convolution(input_tensorflow, filter=conv_filter,
                                              strides=[4, 4], padding='VALID')
        # output_tensorflow_cpu = tf.nn.conv2d(input_tensorflow, filter=conv_filter,
        #                                  strides=[1, 4, 4, 1],padding='VALID')
    elif target_interface == 'pool1':
        output_tensorflow = tf.nn.max_pool(input_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
    elif target_interface == 'pool2':
        output_tensorflow = tf.nn.avg_pool(input_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
    elif target_interface == 'relu1':
        output_tensorflow = tf.nn.relu(input_tensorflow)

    elif target_interface == 'dense1':
        output_tensorflow = tf.layers.dense(input_tensorflow, 1,
                                            kernel_initializer=tf.constant_initializer(0.001))
    elif target_interface == 'sigmoid1':
        output_tensorflow = tf.nn.sigmoid(input_tensorflow)
    elif target_interface == 'tanh1':
        output_tensorflow = tf.tanh(input_tensorflow)
    elif target_interface == 'softmax1':
        output_tensorflow = tf.nn.softmax(input_tensorflow)
    else:
        output_tensorflow = None

    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        with tf.Session() as sess:
            if GPU_mode != 0:
                with tf.device('/cpu:0'):
                    sess.run(tf.global_variables_initializer())
                    sess.run(input_tensorflow)
                    sess.run(output_tensorflow)
                    input_tensorflow_cpu_value = input_tensorflow.eval()
                    output_tensorflow_cpu_value = output_tensorflow.eval()
                    tf.get_variable_scope().reuse_variables()
            if GPU_mode != 2:
                with tf.device('/gpu:0'):
                    sess.run(tf.global_variables_initializer())
                    sess.run(input_tensorflow)
                    sess.run(output_tensorflow)
                    input_tensorflow_gpu_value = input_tensorflow.eval()
                    output_tensorflow_gpu_value = output_tensorflow.eval()
                    tf.get_variable_scope().reuse_variables()

    return output_tensorflow_gpu_value, input_tensorflow_gpu_value, output_tensorflow_cpu_value, input_tensorflow_cpu_value


def tensorflow_compute_multiple(data, target_interface, GPU_mode=1):
    input_tensorflow_gpu_value = None
    output_tensorflow_gpu_value = None
    input_tensorflow_cpu_value = None
    output_tensorflow_cpu_value = None

    # 获取Tensor规模
    tensorflow_shape = [1]
    for shape_element in data.shape:
        tensorflow_shape.append(shape_element)
    # input_tensorflow = tf.reshape(np.array(data).astype(np.float32), tensorflow_shape)
    input_tensorflow = tf.reshape(data, tensorflow_shape)

    # 确定tensorflow接口参数
    if target_interface == 'conv1':
        conv_filter = tf.get_variable('weights_cpu', [11, 11, 3, 1],
                                      initializer=tf.constant_initializer(1),
                                      dtype=tf.float64)
        output_tensorflow = tf.nn.convolution(input_tensorflow, filter=conv_filter,
                                              strides=[4, 4], padding='VALID')
        output_tensorflow = tf.nn.convolution(output_tensorflow, filter=conv_filter,
                                              strides=[4, 4], padding='VALID')
        output_tensorflow = tf.nn.convolution(output_tensorflow, filter=conv_filter,
                                              strides=[4, 4], padding='VALID')
        output_tensorflow = tf.nn.convolution(output_tensorflow, filter=conv_filter,
                                              strides=[4, 4], padding='VALID')
        output_tensorflow = tf.nn.convolution(output_tensorflow, filter=conv_filter,
                                              strides=[4, 4], padding='VALID')
        # output_tensorflow_cpu = tf.nn.conv2d(input_tensorflow, filter=conv_filter,
        #                                  strides=[1, 4, 4, 1],padding='VALID')
    elif target_interface == 'pool1':
        output_tensorflow = tf.nn.max_pool(input_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
        output_tensorflow = tf.nn.max_pool(output_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
        output_tensorflow = tf.nn.max_pool(output_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
        output_tensorflow = tf.nn.max_pool(output_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
        output_tensorflow = tf.nn.max_pool(output_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
    elif target_interface == 'pool2':
        output_tensorflow = tf.nn.avg_pool(input_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
        output_tensorflow = tf.nn.avg_pool(output_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
        output_tensorflow = tf.nn.avg_pool(output_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
        output_tensorflow = tf.nn.avg_pool(output_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
        output_tensorflow = tf.nn.avg_pool(output_tensorflow, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding='VALID')
    elif target_interface == 'relu1':
        output_tensorflow = tf.nn.relu(input_tensorflow)
        output_tensorflow = tf.nn.relu(output_tensorflow)
        output_tensorflow = tf.nn.relu(output_tensorflow)
        output_tensorflow = tf.nn.relu(output_tensorflow)
        output_tensorflow = tf.nn.relu(output_tensorflow)

    elif target_interface == 'dense1':
        output_tensorflow = tf.layers.dense(input_tensorflow, 1,
                                            kernel_initializer=tf.constant_initializer(0.001))
        output_tensorflow = tf.layers.dense(output_tensorflow, 1,
                                            kernel_initializer=tf.constant_initializer(0.001))
        output_tensorflow = tf.layers.dense(output_tensorflow, 1,
                                            kernel_initializer=tf.constant_initializer(0.001))
        output_tensorflow = tf.layers.dense(output_tensorflow, 1,
                                            kernel_initializer=tf.constant_initializer(0.001))
        output_tensorflow = tf.layers.dense(output_tensorflow, 1,
                                            kernel_initializer=tf.constant_initializer(0.001))
    elif target_interface == 'sigmoid1':
        output_tensorflow = tf.nn.sigmoid(input_tensorflow)
        output_tensorflow = tf.nn.sigmoid(output_tensorflow)
        output_tensorflow = tf.nn.sigmoid(output_tensorflow)
        output_tensorflow = tf.nn.sigmoid(output_tensorflow)
        output_tensorflow = tf.nn.sigmoid(output_tensorflow)

    elif target_interface == 'tanh1':
        output_tensorflow = tf.tanh(input_tensorflow)
        output_tensorflow = tf.tanh(output_tensorflow)
        output_tensorflow = tf.tanh(output_tensorflow)
        output_tensorflow = tf.tanh(output_tensorflow)
        output_tensorflow = tf.tanh(output_tensorflow)

    elif target_interface == 'softmax1':
        output_tensorflow = tf.nn.softmax(input_tensorflow)
        output_tensorflow = tf.nn.softmax(output_tensorflow)
        output_tensorflow = tf.nn.softmax(output_tensorflow)
        output_tensorflow = tf.nn.softmax(output_tensorflow)
        output_tensorflow = tf.nn.softmax(output_tensorflow)

    else:
        output_tensorflow = None

    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        with tf.Session() as sess:
            if GPU_mode != 0:
                with tf.device('/cpu:0'):
                    sess.run(tf.global_variables_initializer())
                    sess.run(input_tensorflow)
                    sess.run(output_tensorflow)
                    input_tensorflow_cpu_value = input_tensorflow.eval()
                    output_tensorflow_cpu_value = output_tensorflow.eval()
                    tf.get_variable_scope().reuse_variables()
            if GPU_mode != 2:
                with tf.device('/gpu:0'):
                    sess.run(tf.global_variables_initializer())
                    sess.run(input_tensorflow)
                    sess.run(output_tensorflow)
                    input_tensorflow_gpu_value = input_tensorflow.eval()
                    output_tensorflow_gpu_value = output_tensorflow.eval()
                    tf.get_variable_scope().reuse_variables()

    return output_tensorflow_gpu_value, input_tensorflow_gpu_value, output_tensorflow_cpu_value, input_tensorflow_cpu_value


def tensorflow_compute_combination(data, target_interface, GPU_mode=1):
    input_tensorflow_gpu_value = []
    output_tensorflow_gpu_value = []
    input_tensorflow_cpu_value = []
    output_tensorflow_cpu_value = []
    topology_list = []

    # 获取Tensor规模
    tensorflow_shape = [1]
    for shape_element in data.shape:
        tensorflow_shape.append(shape_element)
    # input_tensorflow = tf.reshape(np.array(data).astype(np.float32), tensorflow_shape)
    input_tensorflow = tf.reshape(data, tensorflow_shape)

    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        with tf.Session() as sess:
            if GPU_mode != 0:
                with tf.device('/cpu:0'):
                    for topology in topologies:
                        if topology.match(target_interface):
                            topology_list.append(topology)
                            data_holder = topology.getData()
                            output_tensorflow = topology.getOp()
                            sess.run(tf.global_variables_initializer())
                            sess.run(input_tensorflow)
                            # sess.run(output_tensorflow, feed_dict={data_holder: input_tensorflow.eval()})
                            output_tensorflow = sess.run(output_tensorflow,
                                                         feed_dict={data_holder: input_tensorflow.eval()})
                            input_tensorflow_cpu_value.append(input_tensorflow.eval())
                            # output_tensorflow_cpu_value.append(output_tensorflow.eval())
                            output_tensorflow_cpu_value.append(output_tensorflow)
                            tf.get_variable_scope().reuse_variables()
            if GPU_mode != 2:
                with tf.device('/gpu:0'):
                    topology_list.clear()
                    for topology in topologies:
                        if topology.match(target_interface):
                            topology_list.append(topology)
                            data_holder = topology.getData()
                            output_tensorflow = topology.getOp()
                            sess.run(tf.global_variables_initializer())
                            sess.run(input_tensorflow)
                            output_tensorflow = sess.run(output_tensorflow,
                                                         feed_dict={data_holder: input_tensorflow.eval()})
                            input_tensorflow_gpu_value.append(input_tensorflow.eval())
                            output_tensorflow_gpu_value.append(output_tensorflow)
                            tf.get_variable_scope().reuse_variables()

    return output_tensorflow_gpu_value, input_tensorflow_gpu_value, output_tensorflow_cpu_value, input_tensorflow_cpu_value, topology_list
