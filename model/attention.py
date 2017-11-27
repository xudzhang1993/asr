from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder
from tensorflow.contrib.seq2seq.python.ops.decoder import dynamic_decode
from tensorflow.contrib.seq2seq.python.ops.helper import TrainingHelper, GreedyEmbeddingHelper
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionWrapper, LuongAttention, BahdanauAttention
from base import BaseModel
from encoder.rnn_encoder import RnnEncoder
from ideas.rnn_score import RnnScoreAttentionWrapper, RnnScoreAttention
from machanism.conv_score import ConvAttention, RealignAttention
from utils.general import get_max_time
from utils import general
from utils.queue import TimitDataSet
from ideas.local_recurrent_attention import LocalAttentionWrapper, LocalGaussAttention


class TrainOutput(namedtuple("TrainOutput", ("update", "loss"))): pass


class EvalOutput(namedtuple("EvalOutput", ("predict_id", "predict_string"))): pass


class Input(namedtuple("Input", ("feature", "target_input", "target_output", "feature_length", "target_length"))): pass


class AttentionModel(BaseModel):
    def __init__(self, hparams, iterator, table, single_batch=False):
        super(AttentionModel, self).__init__(single_batch)
        self.name = self.__class__.__name__ + '_realign'
        self.index_to_string_table = table
        if not single_batch:
            self.inputs = Input(*iterator.get_next())
        else:
            self.inputs = Input(tf.placeholder(tf.float32, [None, None, hparams.feature_dimension]),
                                tf.placeholder(tf.int64, [None, None]), tf.placeholder(tf.int64, [None, None]),
                                tf.placeholder(tf.int64, [None]), tf.placeholder(tf.int64, [None]))
        self.batch_size = tf.size(self.inputs.target_length)

        self.build_graph(hparams)
        self.train_output = TrainOutput(self.update, self.loss)
        self.eval_output = EvalOutput(self.predict_id, self.predict_string)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
        self.target_string = table.lookup(self.inputs.target_output)
        merged = tf.summary.merge_all()
        self.summary = merged

    def build_graph(self, hparams):
        encoder_ouputs, encoder_final_state = self.build_encoder(hparams)
        self.build_decoder(encoder_ouputs, encoder_final_state, hparams)

    def build_encoder(self, hparams):
        encoder = RnnEncoder(hparams)
        return encoder.encode(self.inputs, self.batch_size)

    def build_decoder(self, encoder_outputs, encoder_final_state, hparams):
        decoder_num_hidden = hparams.decoder_num_hidden
        num_alpha = hparams.num_alpha
        with tf.variable_scope("decoder"):
            cell = tf.nn.rnn_cell.GRUCell(decoder_num_hidden)
            cell = AttentionWrapper(cell, BahdanauAttention(decoder_num_hidden, encoder_outputs,
                                                       memory_sequence_length=self.inputs.feature_length,normalize=True
                                                       ),
                                   initial_cell_state=encoder_final_state, output_attention=True,
                                   attention_layer_size=decoder_num_hidden, alignment_history=True)
            #cell = LocalAttentionWrapper(cell, LocalGaussAttention(decoder_num_hidden, encoder_outputs,
            #                                                       self.inputs.feature_length), decoder_num_hidden,
            #                             initial_cell_state=encoder_final_state)
            dense_layer = Dense(num_alpha, dtype=tf.float32, use_bias=False)
            with tf.variable_scope("train"):
                target_input = tf.one_hot(self.inputs.target_input, hparams.num_alpha)
                target_output = tf.one_hot(self.inputs.target_output, hparams.num_alpha)
                helper = TrainingHelper(target_input,
                                        tf.cast(self.inputs.target_length, tf.int32))
                decoder = BasicDecoder(cell, helper, cell.zero_state(self.batch_size, tf.float32), dense_layer)
                decoder_outputs, final_state, _1 = dynamic_decode(decoder, impute_finished=True)
                tf.summary.image('alignment_history', general.create_alignment_image(final_state))
                logits = decoder_outputs.rnn_output
                max_time = get_max_time(target_output, 1)
                target_weights = tf.sequence_mask(
                    self.inputs.target_length, max_time, dtype=logits.dtype)
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=target_output,
                                                            logits=logits) * target_weights)
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), 1)
                self.update = tf.train.AdamOptimizer().apply_gradients(zip(grads, tf.trainable_variables()))
            with tf.variable_scope("eval"):
                def embedding(ids):
                    vec = tf.one_hot(ids, num_alpha, dtype=tf.float32)
                    return vec

                start_tokens = tf.fill([self.batch_size], hparams.sos_id)
                end_tokens = hparams.eos_id
                greedy_helper = GreedyEmbeddingHelper(embedding, start_tokens, end_tokens)
                pre_decoder = BasicDecoder(cell, greedy_helper, cell.zero_state(self.batch_size, tf.float32),
                                           dense_layer)
                pre_decoder_outputs, _, _1 = dynamic_decode(pre_decoder, impute_finished=True, maximum_iterations=50)
                self.predict_id = tf.cast(pre_decoder_outputs.sample_id, tf.int64)
                self.predict_string = self.index_to_string_table.lookup(self.predict_id)
