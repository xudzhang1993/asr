from model.base import BaseEncoder
import tensorflow as tf


class RnnEncoder(BaseEncoder):
    def __init__(self, hparams):
        super(RnnEncoder, self).__init__()
        self.encoder_num_layers = hparams.encoder_num_layers
        self.encoder_num_hidden = hparams.encoder_num_hidden

    def encode(self, inputs, batch_size):
        outputs = inputs.feature
        for i in range(self.encoder_num_layers):
            with tf.variable_scope("encoder_layer_" + str(i)):
                outputs, states = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(self.encoder_num_hidden),
                                                    outputs,
                                                    sequence_length=inputs.feature_length,
                                                    dtype=tf.float32)
        return outputs, states
