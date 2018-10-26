此版本是在改进版的基础上增加TensorBoard可视化功能。
- mnist_train.py文件中，将完成类似功能的计算放到了由tf.name_scope函数生成上下文管理器中。这样TensorBoard可以将这些节点有效的合并，从而突出神经网络的整体结构。
- 因为在mnist_inferenece.py程序中已经使用了tf.variable_scope来管理变量的命名空间，所以这里不需要再做调整。
- 运行完这段代码之后在终端运行 tensorboard --logdir=log/。然后在浏览器输入localhost:6006打开。
- monitor.py是一个可以单独运行的文件，含有监控指标可视化功能。