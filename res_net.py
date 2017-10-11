import tensorflow as tf

NUM_CHANNELS=1
# NUM_LABELS=10

CONV1_DEEP=16
CONV1_SIZE_row=1
CONV1_SIZE_col=3

CONV2_DEEP=16
CONV2_SIZE_row=1
CONV2_SIZE_col=3

CONV3_DEEP=16
CONV3_SIZE_row=1
CONV3_SIZE_col=3

FC_SIZE=1024

OUTPUT_NODE=1

def model(input_tensor,phase,IMAGE_SIZE2):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE_row, CONV1_SIZE_col, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.1))
        conv1_bn_input = tf.contrib.layers.batch_norm(input_tensor,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn_input')
        conv1 = tf.nn.conv2d(conv1_bn_input, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv1_bn=tf.contrib.layers.batch_norm(tf.nn.bias_add(conv1, conv1_biases),
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')
        relu1 = tf.nn.relu(conv1_bn)

    with tf.variable_scope('layer1-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE_row, CONV2_SIZE_col, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv2_bn = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv2, conv2_biases),
                                                center=True, scale=True,
                                                is_training=phase,
                                                scope='bn')
        relu2 = tf.nn.relu(conv2_bn)


    with tf.variable_scope('layer1-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE_row, CONV3_SIZE_col, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.1))

        conv3 = tf.nn.conv2d(relu2, conv3_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv3_bn = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv3, conv3_biases),
                                                center=True, scale=True,
                                                is_training=phase,
                                                scope='bn')
        relu3 = tf.nn.relu(conv3_bn)
    with tf.variable_scope('layer1-shortcut'):
        weights = tf.get_variable("weight", [CONV3_SIZE_row, CONV3_SIZE_col, NUM_CHANNELS, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("bias", [CONV3_DEEP],
                                       initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding="VALID")
        shortcut_y=tf.contrib.layers.batch_norm(tf.nn.bias_add(conv, biases),
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn_input')
        padded_input = tf.pad(relu3, [[0, 0], [0, 0], [2, 2], [0,0]])
        # print (padded_input.get_shape())
        # print (relu3.get_shape())

        shortcut=padded_input+shortcut_y     #(?, 1, 12, 64)




    with tf.variable_scope('layer2-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE_row, CONV1_SIZE_col, CONV3_DEEP, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.1))
        conv1_bn_input = tf.contrib.layers.batch_norm(shortcut,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn_input')
        conv1 = tf.nn.conv2d(conv1_bn_input, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv1_bn=tf.contrib.layers.batch_norm(tf.nn.bias_add(conv1, conv1_biases),
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')
        relu1 = tf.nn.relu(conv1_bn)

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE_row, CONV2_SIZE_col, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv2_bn = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv2, conv2_biases),
                                                center=True, scale=True,
                                                is_training=phase,
                                                scope='bn')
        relu2 = tf.nn.relu(conv2_bn)


    with tf.variable_scope('layer2-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE_row, CONV3_SIZE_col, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.1))

        conv3 = tf.nn.conv2d(relu2, conv3_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv3_bn = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv3, conv3_biases),
                                                center=True, scale=True,
                                                is_training=phase,
                                                scope='bn')
        relu3 = tf.nn.relu(conv3_bn)

    with tf.variable_scope('layer2-shortcut'):
        # weights = tf.get_variable("weight", [CONV3_SIZE_row, CONV3_SIZE_col, NUM_CHANNELS, CONV3_DEEP],
        #                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        # biases = tf.get_variable("bias", [CONV3_DEEP],
        #                          initializer=tf.constant_initializer(0.1))
        # conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 2, 1], padding="VALID")
        # shortcut_y = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv, biases),
        #                                           center=True, scale=True,
        #                                           is_training=phase,
        #                                           scope='bn_input')
        padded_input = tf.pad(relu3, [[0, 0], [0, 0], [3, 3], [0, 0]])

        shortcut1 = shortcut + padded_input  # (?, 1, 12, 64)

        # print (shortcut.get_shape())
        # print (padded_input.get_shape())


    with tf.variable_scope('layer3-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE_row, CONV1_SIZE_col, CONV3_DEEP, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.1))
        conv1_bn_input = tf.contrib.layers.batch_norm(shortcut1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn_input')
        conv1 = tf.nn.conv2d(conv1_bn_input, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv1_bn=tf.contrib.layers.batch_norm(tf.nn.bias_add(conv1, conv1_biases),
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')
        relu1 = tf.nn.relu(conv1_bn)

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE_row, CONV2_SIZE_col, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv2_bn = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv2, conv2_biases),
                                                center=True, scale=True,
                                                is_training=phase,
                                                scope='bn')
        relu2 = tf.nn.relu(conv2_bn)


    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE_row, CONV3_SIZE_col, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.1))

        conv3 = tf.nn.conv2d(relu2, conv3_weights, strides=[1, 1, 1, 1], padding="VALID")
        conv3_bn = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv3, conv3_biases),
                                                center=True, scale=True,
                                                is_training=phase,
                                                scope='bn')
        relu3 = tf.nn.relu(conv3_bn)

    with tf.variable_scope('layer3-shortcut'):
        # weights = tf.get_variable("weight", [CONV3_SIZE_row, CONV3_SIZE_col, NUM_CHANNELS, CONV3_DEEP],
        #                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        # biases = tf.get_variable("bias", [CONV3_DEEP],
        #                          initializer=tf.constant_initializer(0.1))
        # conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 2, 1], padding="VALID")
        # shortcut_y = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv, biases),
        #                                           center=True, scale=True,
        #                                           is_training=phase,
        #                                           scope='bn_input')
        padded_input = tf.pad(relu3, [[0, 0], [0, 0], [3, 3], [0, 0]])

        shortcut2 = shortcut1 + padded_input  # (?, 1, 12, 64)

        # print (shortcut1.get_shape())#(?, 1, 539, 16)
        # print (shortcut2.get_shape())

    with tf.variable_scope('layer4-global_pooling'):
        fc2_weights = tf.get_variable("weights", [CONV3_DEEP, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))

        globalpool = tf.nn.avg_pool(shortcut2, ksize=[1, 1, shortcut2.get_shape()[2], 1], strides=[1, 1,  shortcut2.get_shape()[2], 1], padding='SAME')
        reshape3=tf.reshape(globalpool, [-1,CONV3_DEEP])
        fc=tf.matmul(reshape3, fc2_weights) + fc2_biases
        logit = tf.nn.sigmoid(tf.matmul(reshape3, fc2_weights) + fc2_biases)


    return fc,logit



