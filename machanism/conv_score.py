import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism
from tensorflow.python.layers import core
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
import math


def _conv_score(processed_query, keys, previous_alignments, normalize):
    """Implements Bahdanau-style (additive) scoring function.

    This attention has two forms.  The first is Bhandanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, set `normalize=True`.

    Args:
      processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      normalize: Whether to normalize the score function.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    with tf.variable_scope("conv_score", reuse=tf.AUTO_REUSE):
        v = variable_scope.get_variable(
        "attention_v", [num_units], dtype=dtype)
        Q = tf.get_variable("Q", [100, 1, num_units])
        previous_alignments = tf.expand_dims(previous_alignments, 2)
        F = tf.nn.conv1d(previous_alignments, Q, stride=1, padding="SAME")
        if normalize:
            # Scalar used in weight normalization
            g = variable_scope.get_variable(
                "attention_g", dtype=dtype,
                initializer=math.sqrt((1. / num_units)))
            # Bias added prior to the nonlinearity
            b = variable_scope.get_variable(
                "attention_b", [num_units], dtype=dtype,
                initializer=init_ops.zeros_initializer())
            # normed_v = g * v / ||v||
            normed_v = g * v * tf.rsqrt(
                tf.reduce_sum(tf.square(v)))
            return tf.reduce_sum(
                normed_v * tf.tanh(keys + processed_query + F + b), [2])
        else:
            return tf.reduce_sum(v * tf.tanh(keys + processed_query ), [2])


class ConvAttention(_BaseAttentionMechanism):
    """Implements Bahdanau-style (additive) attention.

    This attention has two forms.  The first is Bahdanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, construct the object with parameter
    `normalize=True`.
    """

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="ConvAttention"):
        """Construct the Attention mechanism.

        Args:
          num_units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
          normalize: Python boolean.  Whether to normalize the energy term.
          probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
          score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          name: Name to use when creating ops.
        """
        if probability_fn is None:
            probability_fn = tf.nn.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(ConvAttention, self).__init__(
            query_layer=core.Dense(
                num_units, name="query_layer", use_bias=False),
            memory_layer=core.Dense(
                num_units, name="memory_layer", use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name

    def __call__(self, query, previous_alignments):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          previous_alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _conv_score(processed_query, self._keys, previous_alignments, self._normalize)
        alignments = self._probability_fn(score, previous_alignments)
        return alignments


class RealignAttention(_BaseAttentionMechanism):
    """Implements Bahdanau-style (additive) attention.

    This attention has two forms.  The first is Bahdanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, construct the object with parameter
    `normalize=True`.
    """

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="ConvAttention"):
        """Construct the Attention mechanism.

        Args:
          num_units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
          normalize: Python boolean.  Whether to normalize the energy term.
          probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
          score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
          name: Name to use when creating ops.
        """
        if probability_fn is None:
            probability_fn = tf.nn.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(RealignAttention, self).__init__(
            query_layer=core.Dense(
                num_units, name="query_layer", use_bias=False),
            memory_layer=core.Dense(
                num_units, name="memory_layer", use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name

    def __call__(self, query, previous_alignments):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          previous_alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _conv_score(processed_query, self._keys, previous_alignments, self._normalize)
            alignments = self._probability_fn(score, previous_alignments)
            alignments = tf.expand_dims(alignments, -1)
            information_context = tf.reduce_sum(alignments * self._keys, 1)
            score = _conv_score(information_context, self._keys, previous_alignments, self._normalize)
            realignments = self._probability_fn(score, previous_alignments)
        return realignments

