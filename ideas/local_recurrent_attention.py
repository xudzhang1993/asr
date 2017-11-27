from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionWrapper, AttentionWrapperState, \
    BahdanauAttention, \
    _BaseAttentionMechanism
import tensorflow as tf
import collections
from tensorflow.python.layers import core as layers_core
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access

State = collections.namedtuple("State",
                               ("cell_state", "context", "time", "alignments",
                                "alignment_history", "position"))

delta = 60
p = np.arange(0, 4 * delta)
norm = np.exp(-np.square(p - 2 * delta) / (2.0 * np.square(delta)))
norm = tf.constant(norm, tf.float32)
norm = tf.expand_dims(norm, 0)


def get_local_matrix(memory, position, delta=delta):
    ta = tf.TensorArray(tf.float32, 0, dynamic_size=True)
    batch_size = memory.shape[0].value or tf.shape(memory)[0]
    int_position = tf.cast(position, tf.int32)
    def get_slice(i, ta):
        ta = ta.write(i, memory[i][int_position[i] - 2 * delta: int_position[i] + 2 * delta])
        return i + 1, ta
    i = tf.Variable(0, dtype=tf.int32)
    _, ta = tf.while_loop(lambda i, _: i < batch_size, get_slice, (i, ta))
    local_memory = ta.stack()
    return local_memory


def bah_attend(processed_query, keys):
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or array_ops.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = array_ops.expand_dims(processed_query, 1)
    v = tf.get_variable(
        "attention_v", [num_units], dtype=dtype)
    return tf.reduce_sum(v * tf.tanh(keys + processed_query), [2])


def next_position(position, cell_output):
    state_depth = cell_output.shape[1]
    wp = tf.get_variable("wp", [state_depth, state_depth], dtype=tf.float32)
    vp = tf.get_variable("vp", [state_depth], dtype=tf.float32)
    vl = tf.get_variable("vl", [state_depth])
    l = tf.exp(tf.reduce_sum(vl * tf.nn.tanh(tf.matmul(cell_output, wp)), 1))
    delta_p = tf.exp(tf.reduce_sum(vp * tf.nn.tanh(tf.matmul(cell_output, wp)), 1))
    position = delta_p + position
    return position, l


class LocalGaussAttention(_BaseAttentionMechanism):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="LocalGaussAttention"):
        if probability_fn is None:
            probability_fn = tf.nn.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(LocalGaussAttention, self).__init__(
            query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False),
            memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._depth = memory.shape[2].value or tf.shape(memory)[2]
        self._memory_sequence_length = memory_sequence_length
        self._num_units = num_units
        self._memory = self._memory_layer(memory)

    def __call__(self, position, query, delta=delta):
        batch_size = 32
        # position = next_position(position, query, self._values, self._memory_sequence_length)
        # local_memory = get_local_matrix(self._memory, position, delta)
        # alignment_bah = bah_attend(self._query_layer(query), local_memory)
        # alignment_gauss = norm * tf.ones([self._batch_size, 1], tf.float32)
        # alignment = alignment_gauss * alignment_bah
        # expand_alignment = tf.expand_dims(alignment, 1)
        # context = tf.matmul(expand_alignment, local_memory)
        # context = tf.squeeze(context, 1)
        # return position, alignment, context
        """Put attention masks on hidden using hidden_features and query."""
        position, l = next_position(position, query)
        attention_states_windows = []
        D = delta
        attn_fixed_length = 2 * delta + 1
        for i in range(batch_size):
            x = tf.constant(D, dtype=dtypes.float32)
            y = math_ops.cast(position[i], dtype=dtypes.float32)

            def f1(): return tf.constant(0, dtype=dtypes.int32), math_ops.cast(D - position[i] + 1, dtype=dtypes.int32)

            def f2():
                return math_ops.cast(position[i] - D, dtype=dtypes.int32), tf.constant(0, dtype=dtypes.int32)

            def f3(): return self._memory_sequence_length[i], math_ops.cast(
                position[i] + D + 2 - tf.cast(self._memory_sequence_length[i], tf.float32), dtype=dtypes.int32)

            def f4(): return math_ops.cast(position[i] + D + 1, dtype=dtypes.int32), tf.constant(0, dtype=dtypes.int32)

            begin, pre_num = tf.cond(tf.less(x, y), f2, f1)
            end, last_num = tf.cond(y + D + 1 < tf.cast(self._memory_sequence_length[i], tf.float32), f4, f3)
            # num = tf.cond(tf.less(end - begin, d), f5, f6)
            pre_tmp = tf.zeros([pre_num, self._num_units], dtype=dtypes.float32)
            last_tmp = tf.zeros([last_num, self._num_units], dtype=dtypes.float32)
            # tmp = tf.zeros([num, attention_vec_size], dtype=dtypes.float32)
            attention_states_window = math_ops.cast(self._values[i][begin:end], dtype=dtypes.float32)
            attention_states_window = tf.concat([pre_tmp, attention_states_window], 0)
            attention_states_window = tf.concat([attention_states_window, last_tmp], 0)
            attention_states_window = attention_states_window[0: 2 * delta + 1]
            attention_states_window = tf.expand_dims(attention_states_window, 0)
            attention_states_windows.append(attention_states_window)

        attention_states_windows = tf.concat(attention_states_windows, 0)
        attention_states_windows = array_ops.reshape(attention_states_windows,
                                                     [batch_size, attn_fixed_length, self._num_units])
        # print(attention_states_windows.shape)

        # To calculate W1 * hi we use a 1-by-1 convolution, need to reshape before.
        hidden_features = attention_states_windows
        v = variable_scope.get_variable("v", [self._num_units])

        with variable_scope.variable_scope("Attention_l"):
            # w2 * ht
            y = self._query_layer(query)
            y = array_ops.reshape(y, [batch_size, 1, self._num_units])
            # Attention mask is a softmax of v^T * tanh(...).
            s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y),
                                    2)
            ai = nn_ops.softmax(s)
            ai = tf.reshape(ai, [batch_size, attn_fixed_length, 1])
            # print(5,ai.get_shape())

            # do the p_t part
            center = tf.constant(D, dtype=dtypes.float32, shape=[batch_size, 1])
            extent = tf.ones([1, attn_fixed_length], dtype=dtypes.float32)
            center = center * extent
            center = tf.reshape(center, [batch_size, attn_fixed_length, 1])

            pos = [i for i in xrange(attn_fixed_length)]
            pos = tf.reshape(pos, [attn_fixed_length, 1])
            pos = math_ops.cast(pos, dtype=dtypes.float32)
            # print((p_t - pos).get_shape(), "jing")

            value = math_ops.square(center - pos) * 2 / (D * D)
            pre = math_ops.exp(math_ops.negative(value))
            # print(pre.get_shape(),"qiu")
            l = tf.reshape(l, [32,1,1])
            ai = l * ai * pre

            # Now calculate the attention-weighted vector d.
            context = math_ops.reduce_sum(
                ai * hidden_features, 1)
            ai = tf.squeeze(ai)
        return position, ai, context


class LocalAttentionWrapper(rnn_cell_impl.RNNCell):
    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        super(LocalAttentionWrapper, self).__init__()
        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._attention_layer_size = attention_layer_size
        self._attention_layer = layers_core.Dense(self._attention_layer_size, activation=tf.tanh, use_bias=False)
        self._cell_input_fn = (
            lambda inputs, attention: array_ops.concat([inputs, attention], -1))
        self._initial_cell_state = initial_cell_state

    @property
    def output_size(self):
        return self._attention_layer_size

    @property
    def state_size(self):
        return None

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            cell_state = self._initial_cell_state
        return State(
            cell_state=cell_state,
            time=tf.zeros([], dtype=tf.int32),
            alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
            context=tf.zeros([batch_size, self._attention_mechanism._depth], dtype=tf.float32),
            position=tf.zeros([batch_size]) + delta + 0.01,
            alignment_history=tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        )

    def call(self, inputs, state):
        cell_inputs = self._cell_input_fn(inputs, state.context)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        position, alignments, context = self._attention_mechanism(state.position, cell_output)
        next_state = State(next_cell_state, context, state.time + 1, alignments, state.alignment_history, position)
        attention = self._attention_layer(array_ops.concat([cell_output, context], 1))
        return attention, next_state
