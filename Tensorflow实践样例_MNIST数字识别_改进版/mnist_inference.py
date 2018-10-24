# -*- coding: utf-8 -*-
import tensorflow as tf

# 总共3层
INPUT_NODE = 784     # 输入层节点数，像素点数
LAYER1_NODE = 500    # 隐藏层节点数
OUTPUT_NODE = 10     # 输出层节点数，类别数


# 通过tf.get_variable函数来获取变量。在训练神经网络时会创建这些变量；在测试时会通过保存的模型加载这些变量的取值。
# 而且更加方便的是，因为可以在变量加载时将滑动平均变量重命名，所以可以直接通过同样的名字在训练时使用变量自身，而在
# 测试时使用变量的滑动平均值。
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights',
                              shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # tf.truncated_normal_initializer 是将变量初始化为满足正态分布的随机值，参数是均值和标准差

    # 当给出了正则化函数时，将当前的正则化损失加入名字为losses的集合，这是自定义的集合，
    # 不在Tensorflow自动管理的集合列表中。
    if regularizer is not None:
        # tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        # 在这里使用tf.get_variable与tf.Variable没有区别，因为在训练或是测试中，没有在同一个程序中重复调用这个函数。
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        # 输入层到隐藏层使用RELU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 类似的声明第二层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
