# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784     # 输入节点
OUTPUT_NODE = 10     # 输出节点
LAYER1_NODE = 500    # 隐藏层数


def get_weight_variable(shape, regularizer):
    '''
    通过tf.get_variable函数来获取变量。在训练神经网络时会创建这些变量；在测试时会通过保存的模型加载这些变量的取值。
    :param shape:
    :param regularizer:
    :return:
    '''
    weights = tf.get_variable('weights',
                              shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # tf.truncated_normal_initializer 是将变量初始化为满足正态分布的随机值，参数是均值和标准差

    if regularizer is not None:
        # tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
