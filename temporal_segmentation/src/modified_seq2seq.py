from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

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

from modified_static_rnn import modified_static_rnn
def modified_rnn_decoder(decoder_inputs,enc_outputs,
                initial_state,enc_states,#all encoder states
                cell,
                loop_function=None,
                scope=None):
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    states = []
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      states.append(state)
      if loop_function is not None:
        prev = output
  return outputs, states,enc_outputs,enc_states

def modified_tied_rnn_seq2seq(encoder_inputs,
                   decoder_inputs,
                   cell,initial_state=None,
                   loop_function=None,
                   dtype=dtypes.float32,
                   scope=None):
  with variable_scope.variable_scope("combined_tied_rnn_seq2seq"):
    scope = scope or "tied_rnn_seq2seq"
    enc_outputs, enc_state,enc_states = modified_static_rnn(
        cell, encoder_inputs, dtype=dtype, scope=scope,initial_state=initial_state)
    variable_scope.get_variable_scope().reuse_variables()
    return modified_rnn_decoder(
        decoder_inputs,enc_outputs,
        enc_state,enc_states,
        cell,
        loop_function=loop_function,
        scope=scope)