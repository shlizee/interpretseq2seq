from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _should_cache():
  """Returns True if a default caching device should be set, otherwise False."""
  if context.executing_eagerly():
    return False
  # Don't set a caching device when running in a loop, since it is possible that
  # train steps could be wrapped in a tf.while_loop. In that scenario caching
  # prevents forward computations in loop iterations from re-reading the
  # updated weights.
  ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  return control_flow_util.GetContainingWhileContext(ctxt) is None
  
def _is_keras_rnn_cell(rnn_cell):
  return (not isinstance(rnn_cell, rnn_cell_impl.RNNCell)
          and isinstance(rnn_cell, base_layer.Layer)
          and getattr(rnn_cell, "zero_state", None) is None)

def _rnn_step(
    time, sequence_length, min_sequence_length, max_sequence_length,
    zero_output, state, call_cell, state_size, skip_conditionals=False):

  # Convert state to a list for ease of use
  flat_state = nest.flatten(state)
  flat_zero_output = nest.flatten(zero_output)

  # Vector describing which batch entries are finished.
  copy_cond = time >= sequence_length

  def _copy_one_through(output, new_output):
    # TensorArray and scalar get passed through.
    if isinstance(output, tensor_array_ops.TensorArray):
      return new_output
    if output.shape.ndims == 0:
      return new_output
    # Otherwise propagate the old or the new value.
    with ops.colocate_with(new_output):
      return array_ops.where(copy_cond, output, new_output)

  def _copy_some_through(flat_new_output, flat_new_state):
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    flat_new_output = [
        _copy_one_through(zero_output, new_output)
        for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
    flat_new_state = [
        _copy_one_through(state, new_state)
        for state, new_state in zip(flat_state, flat_new_state)]
    return flat_new_output + flat_new_state

  def _maybe_copy_some_through():
    """Run RNN step.  Pass through either no or some past state."""
    new_output, new_state = call_cell()

    nest.assert_same_structure(state, new_state)

    flat_new_state = nest.flatten(new_state)
    flat_new_output = nest.flatten(new_output)
    return control_flow_ops.cond(
        # if t < min_seq_len: calculate and return everything
        time < min_sequence_length, lambda: flat_new_output + flat_new_state,
        # else copy some of it through
        lambda: _copy_some_through(flat_new_output, flat_new_state))

  # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
  # but benefits from removing cond() and its gradient.  We should
  # profile with and without this switch here.
  if skip_conditionals:
    # Instead of using conditionals, perform the selective copy at all time
    # steps.  This is faster when max_seq_len is equal to the number of unrolls
    # (which is typical for dynamic_rnn).
    new_output, new_state = call_cell()
    nest.assert_same_structure(state, new_state)
    new_state = nest.flatten(new_state)
    new_output = nest.flatten(new_output)
    final_output_and_state = _copy_some_through(new_output, new_state)
  else:
    empty_update = lambda: flat_zero_output + flat_state
    final_output_and_state = control_flow_ops.cond(
        # if t >= max_seq_len: copy all state through, output zeros
        time >= max_sequence_length, empty_update,
        # otherwise calculation is required: copy some or all of it through
        _maybe_copy_some_through)

  if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
    raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")
  final_output = final_output_and_state[:len(flat_zero_output)]
  final_state = final_output_and_state[len(flat_zero_output):]

  for output, flat_output in zip(final_output, flat_zero_output):
    output.set_shape(flat_output.get_shape())
  for substate, flat_substate in zip(final_state, flat_state):
    if not isinstance(substate, tensor_array_ops.TensorArray):
      substate.set_shape(flat_substate.get_shape())

  final_output = nest.pack_sequence_as(
      structure=zero_output, flat_sequence=final_output)
  final_state = nest.pack_sequence_as(
      structure=state, flat_sequence=final_state)

  return final_output, final_state

def modified_static_rnn(cell,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
  rnn_cell_impl.assert_like_rnncell("cell", cell)
  if not nest.is_sequence(inputs):
    raise TypeError("inputs must be a sequence")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  states = []
  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "rnn") as varscope:
    if _should_cache():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    # Obtain the first sequence of the input
    first_input = inputs
    while nest.is_sequence(first_input):
      first_input = first_input[0]

    # Temporarily avoid EmbeddingWrapper and seq2seq badness
    # TODO(lukaszkaiser): remove EmbeddingWrapper
    if first_input.get_shape().ndims != 1:

      input_shape = first_input.get_shape().with_rank_at_least(2)
      fixed_batch_size = input_shape[0]

      flat_inputs = nest.flatten(inputs)
      for flat_input in flat_inputs:
        input_shape = flat_input.get_shape().with_rank_at_least(2)
        batch_size, input_size = input_shape[0], input_shape[1:]
        fixed_batch_size.merge_with(batch_size)
        for i, size in enumerate(input_size):
          if size.value is None:
            raise ValueError(
                "Input size (dimension %d of inputs) must be accessible via "
                "shape inference, but saw value None." % i)
    else:
      fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

    if fixed_batch_size.value:
      batch_size = fixed_batch_size.value
    else:
      batch_size = array_ops.shape(first_input)[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, "
                         "dtype must be specified")
      if getattr(cell, "get_initial_state", None) is not None:
        state = cell.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype)
      else:
        state = cell.zero_state(batch_size, dtype)

    if sequence_length is not None:  # Prepare variables
      sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if sequence_length.get_shape().ndims not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size")

      def _create_zero_output(output_size):
        # convert int to TensorShape if necessary
        size = _concat(batch_size, output_size)
        output = array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))
        shape = _concat(fixed_batch_size.value, output_size, static=True)
        output.set_shape(tensor_shape.TensorShape(shape))
        return output

      output_size = cell.output_size
      flat_output_size = nest.flatten(output_size)
      flat_zero_output = tuple(
          _create_zero_output(size) for size in flat_output_size)
      zero_output = nest.pack_sequence_as(
          structure=output_size, flat_sequence=flat_zero_output)

      sequence_length = math_ops.to_int32(sequence_length)
      min_sequence_length = math_ops.reduce_min(sequence_length)
      max_sequence_length = math_ops.reduce_max(sequence_length)

    # Keras RNN cells only accept state as list, even if it's a single tensor.
    is_keras_rnn_cell = _is_keras_rnn_cell(cell)
    if is_keras_rnn_cell and not nest.is_sequence(state):
      state = [state]
    for time, input_ in enumerate(inputs):
      if time > 0:
        varscope.reuse_variables()
      # pylint: disable=cell-var-from-loop
      call_cell = lambda: cell(input_, state)
      # pylint: enable=cell-var-from-loop
      if sequence_length is not None:
        (output, state) = _rnn_step(
            time=time,
            sequence_length=sequence_length,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            zero_output=zero_output,
            state=state,
            call_cell=call_cell,
            state_size=cell.state_size)
      else:
        (output, state) = call_cell()
      outputs.append(output)
      states.append(state)
    # Keras RNN cells only return state as list, even if it's a single tensor.
    if is_keras_rnn_cell and len(state) == 1:
      state = state[0]

    return (outputs, state,states)