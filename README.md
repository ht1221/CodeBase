## Welcome to the codebase of huangting.1221

# `log.py` : 日志、输出相关的代码
### ``def get_logger(log_path)`` : 
输入为log的保存路径，返回一个logging的logger，设定的level为DEBUG(可以根据需求修改为WARNING, INFO等)，方便管理程序不同级别的打印信息，比print好用。

# `models.py` : 构建模型相关的代码
### `conv_block(inputs, cnn_block_config, training=None)`: 
输入卷积块的是格式为``"NHWC"``的常规的CNN的输入，一个卷积块的配置信息，输出是卷积块的feature maps。其中卷积块中允许的计算有：conv2d, norm, activation, max_pool。cnn_block_config的一个例子：
```
{
  "name": "block_1",
  "operations":[
      ("conv2d", [[3, 3, 64, 1, 1, "same"], ]),
      ("norm", "bn"),
      ("activation", tf.nn.relu),
      ("max_pool", (2, 2, 2, 2, "same")),
  ]
}
```

# `save_and_restore_model.py`: 保存和导入模型相关的代码
### `freeze_graph(input_checkpoint_path, output_graph)`:
将保存的checkpoint转换为PB格式，输入是保存好的checkpoint文件和PB格式文件将要保存的路径。除了提供两个路径之外，还需要知道

 1. 输入tensor的名字
 2.  输出tensor的名字
 3. 需要freeze的节点名字(node name) 注意：节点的名字和tensor的名字有不同，一般是不带`:0`的名字