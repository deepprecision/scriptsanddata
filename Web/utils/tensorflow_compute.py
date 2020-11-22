import numpy as np
import tensorflow as tf


def tensorflow_compute_all(im, target_interface, GPU_mode=1, comparFramework=''):
    input_tensorflow_value = None
    output_tensorflow_value = None
    input_tensorflow_cpu_value = None
    output_tensorflow_cpu_value = None
    if GPU_mode != 0:
        with tf.Session() as sess:
            # 获取Tensor规模
            tensorflow_shape = [1]
            for shape_element in im.shape:
                tensorflow_shape.append(shape_element)
            # input_tensorflow = tf.reshape(np.array(data).astype(np.float32), tensorflow_shape)
            input_tensorflow = tf.reshape(im, tensorflow_shape)
            input_tensorflow_cpu_value = input_tensorflow.eval()

            # 确定tensorflow接口参数
            if target_interface == 'conv1':
                conv_filter = tf.convert_to_tensor(np.full((11, 11, 3, 1), 1, np.float64))
                output_tensorflow_cpu = tf.nn.conv2d(input_tensorflow, filter=conv_filter,
                                                     strides=[1, 4, 4, 1], padding='VALID')
            elif target_interface == 'conv2':
                conv_filter = tf.convert_to_tensor(np.full((11, 11, 3, 1), 1, np.float64))
                output_tensorflow_cpu = tf.nn.conv2d(input_tensorflow, filter=conv_filter,
                                                     strides=[1, 4, 4, 1], padding='SAME')
            elif target_interface == 'pool1':
                output_tensorflow_cpu = tf.nn.max_pool(input_tensorflow, ksize=[1, 2, 2, 1],
                                                       strides=[1, 2, 2, 1], padding='VALID')
            elif target_interface == 'pool2':
                output_tensorflow_cpu = tf.nn.avg_pool(input_tensorflow, ksize=[1, 2, 2, 1],
                                                       strides=[1, 2, 2, 1], padding='VALID')
            elif target_interface == 'relu1':
                output_tensorflow_cpu = tf.nn.relu(input_tensorflow)

            elif target_interface == 'dense1':
                output_tensorflow_cpu = tf.layers.dense(input_tensorflow, 1,
                                                        kernel_initializer=tf.constant_initializer(0.001))
            elif target_interface == 'sigmoid1':
                output_tensorflow_cpu = tf.nn.sigmoid(input_tensorflow)
            elif target_interface == 'tanh1':
                output_tensorflow_cpu = tf.tanh(input_tensorflow)
            elif target_interface == 'softmax1':
                output_tensorflow_cpu = tf.nn.softmax(input_tensorflow)
            else:
                output_tensorflow_cpu = None

            # 获取tensorflow结果
            sess.run(tf.global_variables_initializer())
            sess.run(output_tensorflow_cpu)
            output_tensorflow_cpu_value = output_tensorflow_cpu.eval()
    if GPU_mode != 2:
        with tf.Session() as sess:
            # GPU
            # 获取Tensor规模
            tensorflow_shape = [1]
            for shape_element in im.shape:
                tensorflow_shape.append(shape_element)
            # input_tensorflow = tf.reshape(np.array(data).astype(np.float32), tensorflow_shape)
            input_tensorflow = tf.reshape(im, tensorflow_shape)
            input_tensorflow_value = input_tensorflow.eval()

            # 确定tensorflow接口参数
            if target_interface == 'conv1':
                conv_filter = tf.convert_to_tensor(np.full((11, 11, 3, 1), 1, np.float64))
                output_tensorflow = tf.nn.conv2d(input_tensorflow, filter=conv_filter,
                                                 strides=[1, 4, 4, 1], padding='VALID')
            elif target_interface == 'conv2':
                conv_filter = tf.convert_to_tensor(np.full((11, 11, 3, 1), 1, np.float64))
                output_tensorflow = tf.nn.conv2d(input_tensorflow, filter=conv_filter,
                                                 strides=[1, 4, 4, 1], padding='SAME')
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
                                                    kernel_initializer=tf.constant_initializer(1))
            elif target_interface == 'sigmoid1':
                output_tensorflow = tf.nn.sigmoid(input_tensorflow)
            elif target_interface == 'tanh1':
                output_tensorflow = tf.tanh(input_tensorflow)
            elif target_interface == 'softmax1':
                output_tensorflow = tf.nn.softmax(input_tensorflow)
            else:
                output_tensorflow = None

            # 获取tensorflow结果
            sess.run(tf.global_variables_initializer())
            sess.run(output_tensorflow)
            output_tensorflow_value = output_tensorflow.eval()

    return output_tensorflow_value, input_tensorflow_value, output_tensorflow_cpu_value, input_tensorflow_cpu_value
