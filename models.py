# coding: utf-8
''' 构建模型相关的代码 '''

def conv_block(inputs, cnn_block_config, training=None):
        ''' 新版本的cnn block封装方式，配置文件的可读性会比之前更好一些, 没有使用batch_normalization时候不需要使用training这个参数
        以下是一个cnn block的配置信息
        {
            "name": "block_1",
            "operations":[
                ("conv2d", [[3, 3, 64, 1, 1, "same"], ]),
                ("norm", "bn"),
                ("activation", tf.nn.relu),
                ("max_pool", (2, 2, 2, 2, "same")),
            ]
        },
        '''
        for tag in ["name", "operations"]:
            if tag not in cnn_block_config:
                raise ValueError("no {} infomation in the cnn_block_config".format(tag))

        name = cnn_block_config["name"]
        operations = cnn_block_config["operations"]
        with tf.variable_scope(name) as scope:
            # 初始化计算流程,feats在流程中一直作为中间结果存在
            feats = inputs
            # 按照顺序依次对输入进行相应的计算
            for operation in operations:
                op_name, op_param = operation
                if op_name == "conv2d":
                    part_feat_list = []
                    # 同一个cnn block块中每一个类型卷积的输出按照channel进行拼接
                    for kernel in op_param:
                        # (height, width, n_kernels, stride_h, stride_w, pad_mode)
                        part_feats = tf.layers.conv2d(
                        inputs=feats,
                        filters=kernel[2],
                        kernel_size=kernel[:2],
                        strides=kernel[3:5],
                        padding=kernel[-1],
                        )
                        part_feat_list.append(part_feats)
                    feats = tf.concat(part_feat_list, axis=-1)  # channel拼接
                elif op_name == "norm":
                    if op_param == "bn":
                        # batch_normalization 使用默认的参数
                        feats = tf.layers.batch_normalization(feats, training=training)
                    elif op_param == "lrn":
                        # local response normalization 使用统一的一组参数
                        feats = tf.nn.lrn(feats, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=scope.name + 'norm')
                    else:
                        raise ValueError("bad normalization name: {}".format(op_param))
                elif op_name == "activation":
                    feats = op_param(feats)
                elif op_name == "max_pool":
                    # ("max_pool", (2, 2, 2, 2, "same"))
                    feats = tf.layers.max_pooling2d(feats, op_param[:2], strides=op_param[2:4], padding=op_param[-1])
                else:
                    raise ValueError("bad operation name: {}".format(op_name))

        return feats
