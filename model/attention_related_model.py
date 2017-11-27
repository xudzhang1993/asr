from model.attention import AttentionModel
from encoder.conv_encoder import ConvEncoder
from encoder.rnn_encoder import RnnEncoder
import tensorflow as tf

class ConvModel(AttentionModel):
    def build_encoder(self, hparams):
        encoder = ConvEncoder(hparams)
        return encoder.encode(self.inputs, self.batch_size), None

class EncoderConvModel(AttentionModel):

    def build_encoder(self, hparams):
        encoder = RnnEncoder(hparams)
        shape = [5, hparams.en, hparams.encoder_num_hidden]
        filter = tf.get_variable("filter", shape, tf.float32)
        encoder_outputs, final_state = encoder.encode(self.inputs, self.batch_size)
        encoder_outputs = tf.nn.conv1d(encoder_outputs, filter, 1, "SAME")
        return encoder_outputs, final_state

