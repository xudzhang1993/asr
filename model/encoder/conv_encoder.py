from model.base import BaseEncoder
import tensorflow as tf

class ConvEncoder(BaseEncoder):
    def __init__(self, hparams):
        super(ConvEncoder, self).__init__()
        self.filter_width = hparams.filter_width
        self.filter_channels = hparams.filter_channels
        self.filter_stride = hparams.filter_stride
        self.encoder_num_layers = hparams.encoder_num_layers
        self.feature_dimension = hparams.feature_dimension

    def encode(self, inputs, batch_size):
        shape_0 = [self.filter_width, self.feature_dimension, self.filter_channels]
        shape_i = [self.filter_width, self.filter_channels, self.filter_channels]
        output = inputs.feature
        for i in range(self.encoder_num_layers):
            with tf.variable_scope("encoder_layer_" + str(i)):
                if i == 0:
                    shape = shape_0
                else:
                    shape = shape_i
                # filters = tf.get_variable("filter", shape, tf.float32)
                output = conv1d_weightnorm(output, i, self.filter_channels, self.filter_width)
        return output

def res(inputs, indims, outdims):
    shape = tf.shape(inputs)
    resw = tf.get_variable("w", [indims, outdims])
    outputs = tf.reshape(inputs, [-1, shape[-1]])
    outputs = tf.matmul(outputs, resw)
    return tf.reshape(outputs,(shape[0], -1, outdims))


def linear_mapping_weightnorm(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()  # static shape. may has None
        input_shape_tensor = tf.shape(inputs)
        # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
        V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                dropout * 1.0 / int(input_shape[-1]))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(),
                            trainable=True)  # weightnorm bias is init zero

        assert len(input_shape) == 3
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
        # inputs = tf.matmul(inputs, V)    # x*v

        scaler = tf.div(g, tf.norm(V, axis=0))  # g/2-norm(v)
        inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])  # x*v g/2-norm(v) + b

        return inputs


def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,
                      var_scope_name="conv_layer"):  # padding should take attention

    with tf.variable_scope("conv_layer_" + str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(
                                4.0 * dropout / (kernel_size * in_dim))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
        return inputs