from ideas.rnn_conv import dynamic_rnn_conv, RnnConvKernel
from model.base import BaseEncoder
import tensorflow as tf


class RnnConvEncoder(BaseEncoder):

    def __init__(self, hparams):
        super(BaseEncoder, self).__init__()
        self.encoder_num_layers = hparams.encoder_num_layers
        self.encoder_num_hidden = hparams.encoder_num_hidden

    def encode(self, inputs, batch_size):
        outputs = inputs
        for i in range(self.encoder_num_layers):
            with tf.variable_scope("encoder_layer_" + str(i)):
                kernel = RnnConvKernel(5, tf.nn.rnn_cell.GRUCell(self.encoder_num_hidden))
                outputs = dynamic_rnn_conv(outputs, None, kernel, 1)
        return outputs


def encode(inputs, hparams):
    with tf.variable_scope("encoder_layer1"):
        kernel1 = RnnConvKernel(5, tf.nn.rnn_cell.GRUCell(hparams.encoder_num_hidden))
        outputs = dynamic_rnn_conv(inputs, None, kernel1,1)
    with tf.variable_scope("encoder_layer2"):
        kernel2 = RnnConvKernel(5, tf.nn.rnn_cell.GRUCell(hparams.encoder_num_hidden))
        outputs = dynamic_rnn_conv(outputs, None, kernel2, 1)
    return outputs
