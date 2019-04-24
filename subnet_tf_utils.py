import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name) as scope:
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def linear1d(inputlin, inputdim, outputdim, name="linear1d", std=0.02, mn=0.0):
    with tf.variable_scope(name) as scope:
        weight = tf.get_variable("weight", [inputdim, outputdim])
        bias = tf.get_variable("bias", [outputdim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        return tf.matmul(inputlin, weight) + bias


def general_conv2d(inputconv, output_dim=64, filter_height=4, filter_width=4, stride_height=2, stride_width=2,
                   stddev=0.02, padding="SAME", name="conv2d", do_norm=True, norm_type='instance_norm', do_relu=True,
                   relufactor=0, is_training=True):
    with tf.variable_scope(name) as scope:
        conv = tf.contrib.layers.conv2d(inputconv, output_dim, [filter_width, filter_height],
                                        [stride_width, stride_height], padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            if norm_type == 'instance_norm':
                conv = instance_norm(conv)
            elif norm_type == 'batch_norm':
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=is_training, updates_collections=None,
                                                    epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if relufactor == 0:
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_deconv2d(inputconv, output_dim=64, filter_height=4, filter_width=4, stride_height=2, stride_width=2,
                     stddev=0.02, padding="SAME", name="deconv2d", do_norm=True, norm_type='instance_norm',
                     do_relu=True,
                     relufactor=0, do_tanh=False, is_training=True):
    with tf.variable_scope(name) as scope:
        conv = tf.contrib.layers.conv2d_transpose(inputconv, output_dim, [filter_height, filter_width],
                                                  [stride_height, stride_width], padding, activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                  biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            if norm_type == 'instance_norm':
                conv = instance_norm(conv)
            elif norm_type == 'batch_norm':
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=is_training, updates_collections=None,
                                                    epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        if do_tanh:
            conv = tf.nn.tanh(conv, "tanh")

        return conv


def generative_cnn_encoder(inputs, is_training=True, drop_keep_prob=0.5, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
        o_c1 = general_conv2d(inputs, 32, is_training=is_training, name="CNN_ENC_1")
        o_c2 = general_conv2d(o_c1, 64, is_training=is_training, name="CNN_ENC_2")
        o_c3 = general_conv2d(o_c2, 128, is_training=is_training, name="CNN_ENC_3")
        o_c4 = general_conv2d(o_c3, 256, is_training=is_training, name="CNN_ENC_4")
        o_c5 = general_conv2d(o_c4, 256, is_training=is_training, name="CNN_ENC_5")
        o_c5 = tf.reshape(o_c5, (-1, 256 * 7 * 7))
        o_c6 = linear1d(o_c5, 256 * 7 * 7, 512, name='CNN_ENC_FC')
        # TODO: here?
        # o_c6 = tf.cond(is_training, lambda: tf.nn.dropout(o_c6, 0.5), lambda: o_c6)
        o_c6 = tf.nn.dropout(o_c6, drop_keep_prob)

        return o_c6


def generative_cnn_decoder(inputs, is_training=True, drop_keep_prob=0.5, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
        o_d1 = linear1d(inputs, 128, 256 * 7 * 7, name='CNN_DEC_FC')
        # o_d1 = tf.cond(is_training, lambda: tf.nn.dropout(o_d1, 0.5), lambda: o_d1)
        o_d1 = tf.nn.dropout(o_d1, drop_keep_prob)
        o_d1 = tf.reshape(o_d1, [-1, 7, 7, 256])
        o_d2 = general_deconv2d(o_d1, 256, is_training=is_training, name="CNN_DEC_1")
        o_d3 = general_deconv2d(o_d2, 128, is_training=is_training, name="CNN_DEC_2")
        o_d4 = general_deconv2d(o_d3, 64, is_training=is_training, name="CNN_DEC_3")
        o_d5 = general_deconv2d(o_d4, 32, is_training=is_training, name="CNN_DEC_4")
        # TODO: here?
        # o_d6 = general_deconv2d(o_d5, 3, is_training=is_training, name="CNN_DEC_5", do_relu=False, do_tanh=True)
        o_d6 = general_deconv2d(o_d5, 3, name="CNN_DEC_5", do_norm=False, do_relu=False, do_tanh=True)

        return o_d6
