import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism, AttentionWrapper, AttentionWrapperState, ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.layers import core as layers_core


def rnn_score(initial_state, keys, cell, feature_length):
    outputs, final_state = tf.nn.dynamic_rnn(cell, keys, feature_length, initial_state)
    num_units = keys.shape[2].value or tf.array_ops.shape(keys)[2]
    v = tf.get_variable("v", shape=[num_units])
    score = tf.reduce_sum(v * outputs, 2)
    return score, final_state

class RnnScoreAttention(_BaseAttentionMechanism):
  """Implements Luong-style (multiplicative) attention scoring.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.

  To enable the second form, construct the object with parameter
  `scale=True`.
  """

  def __init__(self,
               num_units,
               memory,
               cell,
               memory_sequence_length=None,
               scale=False,
               probability_fn=None,
               score_mask_value=float("-inf"),
               name="LuongAttention"):
    """Construct the AttentionMechanism mechanism.

    Args:
      num_units: The depth of the attention mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      scale: Python boolean.  Whether to scale the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(RnnScoreAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name
    self._memory_sequence_length = memory_sequence_length
    self._cell = cell

  def __call__(self, initial_state, previous_alignments):
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
    with tf.variable_scope(None, "rnn_score_attention", [initial_state]):
        score, final_state = rnn_score(initial_state, self._keys, self._cell, self._memory_sequence_length)
    alignments = self._probability_fn(score, previous_alignments)
    return alignments, final_state

def _compute_attention(attention_mechanism, initial_state, previous_alignments,
                       attention_layer):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments, final_state = attention_mechanism(
      initial_state, previous_alignments=previous_alignments)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = tf.expand_dims(alignments, 1)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  context = tf.matmul(expanded_alignments, attention_mechanism.values)
  context = tf.squeeze(context, [1])

  if attention_layer is not None:
    attention = attention_layer(context)
  else:
    attention = context

  return attention, alignments, final_state


class RnnScoreAttentionWrapper(AttentionWrapper):
    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).

        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.

        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        _, initial_state = self._cell(cell_inputs, state.cell_state)
        cell_state = state.cell_state


        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_cell_state = _compute_attention(
                attention_mechanism, initial_state, previous_alignments[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)

        attention = tf.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(all_histories))

        return attention, next_state
