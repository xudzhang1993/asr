
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
# import math

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def attention(query, ):
    """Put attention masks on hidden using hidden_features and query."""
    ds = []  # Results of attention reads will be stored here.
    if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
            ndims = q.get_shape().ndims
            if ndims:
                assert ndims == 2
        query = array_ops.concat(query_list, 1)

    with variable_scope.variable_scope("Attention_%d" % a, dtype=dtype):
    attention_vec_size = attn_size  # Size of query vectors for attention.
    # to calucate wp * ht
    v_p = variable_scope.get_variable("AttnV_p%d" % a, [attention_vec_size])
    qiu = linear(query, attention_vec_size, True)
    qiu = array_ops.reshape(qiu, [batch_size, 1, 1, attention_vec_size])
    tan_v = math_ops.reduce_sum(v_p * math_ops.tanh(qiu),
                                [2, 3])
    # print(tan_v.get_shape())
    pt_sig = math_ops.sigmoid(tan_v)
    # print(pt_sig.get_shape())
    p = attn_length * pt_sig
    # print(p.get_shape())
    # p_t = (array_ops.reshape(p, [-1, attn_length]))
    p_t = math_ops.cast(p, dtype=dtypes.int32)
    p_t = math_ops.cast(p_t, dtype=dtypes.float32)
    # print(p_t.get_shape())
    # print(4)
    # p_t=tf.convert_to_tensor(p_t)

    # print(p_t.shape, attention_states.shape)

    # set a window
    p_t = array_ops.reshape(p_t, [batch_size, ])
    attention_states_windows = []
    D = attn_local_D
    for i in range(attention_states.shape[0]):
        x = tf.constant(D, dtype=dtypes.float32)
        y = math_ops.cast(p_t[i], dtype=dtypes.float32)
        z = tf.constant(attn_length, dtype=dtypes.float32)

        def f1(): return tf.constant(0, dtype=dtypes.int32), math_ops.cast(D - p_t[i], dtype=dtypes.int32)

        def f2():
            return math_ops.cast(p_t[i] - D, dtype=dtypes.int32), tf.constant(0, dtype=dtypes.int32)

        def f3(): return tf.constant(attn_length, dtype=dtypes.int32), math_ops.cast(
            p_t[i] + D + 1 - attn_length, dtype=dtypes.int32)

        def f4(): return math_ops.cast(p_t[i] + D + 1, dtype=dtypes.int32), tf.constant(0, dtype=dtypes.int32)

        begin, pre_num = tf.cond(tf.less(x, y), f2, f1)
        end, last_num = tf.cond(tf.less(y + D + 1, z), f4, f3)

        d = tf.constant(attn_fixed_length, dtype=dtypes.int32)
        # num = tf.cond(tf.less(end - begin, d), f5, f6)
        pre_tmp = tf.zeros([pre_num, attention_vec_size], dtype=dtypes.float32)
        last_tmp = tf.zeros([last_num, attention_vec_size], dtype=dtypes.float32)
        # tmp = tf.zeros([num, attention_vec_size], dtype=dtypes.float32)
        attention_states_window = math_ops.cast(attention_states[i][begin:end], dtype=dtypes.float32)
        attention_states_window = tf.concat([pre_tmp, attention_states_window], 0)
        attention_states_window = tf.concat([attention_states_window, last_tmp], 0)
        attention_states_window = tf.expand_dims(attention_states_window, 0)
        attention_states_windows.append(attention_states_window)

    attention_states_windows = tf.concat(attention_states_windows, 0)
    attention_states_windows = array_ops.reshape(attention_states_windows,
                                                 [batch_size, attn_fixed_length, attention_vec_size])
    # print(attention_states_windows.shape)

    # To calculate W1 * hi we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(attention_states_windows,
                               [batch_size, attn_fixed_length, 1, attn_size])
    k = variable_scope.get_variable("AttnW_%d" % a,
                                    [1, 1, attn_size, attention_vec_size])
    hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
    v = variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size])

    with variable_scope.variable_scope("Attention_l_%d" % a, dtype=dtype):
        # w2 * ht
        y = linear(query, attention_vec_size, True)
        y = array_ops.reshape(y, [batch_size, 1, 1, attention_vec_size])
        # Attention mask is a softmax of v^T * tanh(...).
        s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y),
                                [2, 3])
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
        ai = ai * pre

        # Now calculate the attention-weighted vector d.
        d = math_ops.reduce_sum(
            array_ops.reshape(ai, [batch_size, attn_fixed_length, 1, 1]) * hidden, [1, 2])
        ds.append(array_ops.reshape(d, [batch_size, attn_size]))
    return ds

