from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import random

"""
Some utils for Quaternion functions and layers.

References:

https://arxiv.org/pdf/1806.04418.pdf
https://arxiv.org/pdf/1806.07789.pdf

https://github.com/Orkis-Research/light-Recurrent-Neural-Networks
https://github.com/Orkis-Research/light-Convolutional-Neural-Networks-for-End-to-End-Automatic-Speech-Recognition

Some functions are direct ports from the Pytorch library.
"""

def make_quarternion_mul(kernel, concat_dim=0):
	r, i, j, k = tf.split(kernel, 4, axis=-1)
	r2 = tf.concat([r, -i, -j, -k], axis=-1)	# 0, 1, 2, 3
	i2 = tf.concat([i, r, -k, j], axis=-1)	# 1, 0, 3, 2
	j2 = tf.concat([j, k, r, -i], axis=-1)	# 2, 3, 0, 1
	k2 = tf.concat([k, -j, i, r],axis=-1)	# 3, 2, 1, 0
	hamilton = tf.concat([r2, i2, j2, k2], axis=concat_dim)
	return hamilton

def get_r(x, a=1):
	return tf.split(x, 4, axis=a)[0]

def get_i(x, a=1):
	return tf.split(x, 4, axis=a)[1]

def get_j(x, a=1):
	return tf.split(x, 4, axis=a)[2]

def get_k(x, a=1):
	return tf.split(x, 4, axis=a)[3]

def quarternion_attention(a, b):
	""" Performs dot product attention between two quarternion sequences.

	a = bsz x al x dim
	b = bsz x bl x dim

	following:
	(rr' - xx' - yy' - zz')  +
		(rx' + xr' + yz' - zy')i +
		(ry' - xz' + yr' + zx')j +
		(rz' + xy' - yx' + zr')k +

	the output should be one attention matrix for each component (r,i,j,k)
	"""
	print("light Attention!")
	print(a)
	print(b)
	al, bl = tf.shape(a)[2], tf.shape(b)[2]

	ar, ax, ay, az = tf.split(a, 4, axis=-1)
	br, bx, by, bz = tf.split(b, 4, axis=-1)
	r = tf.matmul(ar, br, transpose_b=True) - tf.matmul(ax, bx, transpose_b=True) - tf.matmul(ay, by, transpose_b=True) - tf.matmul(az, bz, transpose_b=True)
	i = tf.matmul(ar, bx, transpose_b=True) + tf.matmul(ax, br, transpose_b=True) + tf.matmul(ay, bz, transpose_b=True) - tf.matmul(az, by, transpose_b=True)
	j = tf.matmul(ar, by, transpose_b=True) - tf.matmul(ax, bz, transpose_b=True) + tf.matmul(ay, br, transpose_b=True) + tf.matmul(az, bx, transpose_b=True)
	k = tf.matmul(ar, bz, transpose_b=True) + tf.matmul(ax, by, transpose_b=True) - tf.matmul(ay, bx, transpose_b=True) + tf.matmul(az, br, transpose_b=True)
	return [r, i, j, k]

def quarternion_dot_product_att(a, b):
	""" Wrapper for two sequences
	"""
	al = tf.shape(a)[1]
	bl = tf.shape(b)[1]
	# print(a)
	d = a.get_shape().as_list()[2]
	bsz = tf.shape(b)[0]
	a = tf.reshape(a, [-1, d])
	a = tf.tile(a, [bl, 1])
	b = tf.reshape(b, [-1, d])
	b = tf.tile(b, [al, 1])
	att = quarternion_dot(a, b)
	att = tf.reshape(att, [bsz, -1, al * bl])
	att = tf.reduce_sum(att, 1)
	return tf.reshape(att, [-1, al * bl])

def quarternion_dot_3d(q0, q1):
	d = q0.get_shape().as_list()[2]
	sq = tf.shape(q0)[1]
	q0 = tf.reshape(q0, [-1, d])
	q1 = tf.reshape(q1, [-1, d])
	out = quarternion_dot(q0, q1)
	return tf.reshape(out, [-1, sq, d])

def quarternion_dot(q0, q1):
	""" Quarternion product between 2 quarternions

	returns same shape and acts like element-wise quarternion mul
	"""
	q1_r = get_r(q1)
	q1_i = get_i(q1)
	q1_j = get_j(q1)
	q1_k = get_k(q1)

	r_base = tf.multiply(q0, q1)
	r = get_r(r_base) - get_i(r_base) - get_j(r_base) - get_k(r_base)

	i_base = tf.multiply(q0, tf.concat([q1_i, q1_r, q1_k, q1_j], 1))
	i = get_r(i_base) + get_i(i_base) + get_j(i_base) - get_k(i_base)

	j_base = tf.multiply(q0, tf.concat([q1_j, q1_k, q1_r, q1_i], 1))
	j = get_r(j_base) - get_i(j_base) + get_j(j_base) + get_k(j_base)

	k_base = tf.multiply(q0, tf.concat([q1_k, q1_j, q1_i, q1_r], 1))
	k = get_r(k_base) + get_i(k_base) - get_j(k_base) + get_k(k_base)

	return tf.concat([r, i, j, k], 1)

def quarternion_concat(x, axis):
	""" Helpful if we have 2 quarternions in [r,i,j,k].
	We can't simply concat them as it would mess the components.
	So in this case, we extract each component and concat them individually.
	"""
	output = [[] for i in range(4)]
	for _x in x:
		sp = tf.split(_x, 4, axis=axis)
		for i in range(4):
			output[i].append(sp[i])

	final = []
	for o in output:
		o = tf.concat(o, axis)
		final.append(o)

	return tf.concat(final, axis)

def quarternion_ffn_3d(x, dim, name='', init=None,
				num_layers=1, activation=None, reuse=None):
	""" Quarternion Feed-forward layers to 3D input [bsz x seq_len x dim]
	returns same shape tensor with new projected dimension.
	"""
	print("QFFN layer..")
	_d = x.get_shape().as_list()[2]
	sq = tf.shape(x)[1]
	x = tf.reshape(x, [-1, _d])
	x = quarternion_ffn(x, dim, name=name, init=init,
						num_layers=num_layers,
						activation=activation,reuse=reuse)
	x = tf.reshape(x, [-1, sq, dim])
	return x

def quarternion_ffn(x, dim, name='', init=None,
				num_layers=1, activation=None, reuse=None):
	""" Implements quarternion feed-forward layer

	x is [bsz x features] tensor
	"""
	if(init is None):
		init = tf.contrib.layers.xavier_initializer()
		# init = q_xavier_initializer()
	input_dim = x.get_shape().as_list()[1] // 4
	with tf.variable_scope('Q{}'.format(name), reuse=reuse) as scope:
		kernel = tf.get_variable('quarternion', [input_dim, dim], initializer=init)
		hamilton = make_quarternion_mul(kernel)
		output = tf.matmul(x, hamilton)
		if(activation):
			output = activation(output)
		return output

def hamilton_product(x, kernel):
	h = make_quarternion_mul(kernel)
	output = tf.matmul(x, h)
	return output

# Code beyond this line is not used in this repository.

class QuarternionRNN(tf.nn.rnn_cell.RNNCell):

	def __init__(self, input_dim, output_dim,
					initializer=None, name='', reuse=None):
		""" Rough implementation (need double-check)
		from the Quarternion RNN paper. For now, works decently.
		"""
		self.dim = output_dim
		with tf.variable_scope("QuartRNN{}".format(name), reuse=reuse) as scope:
			if(initializer is None):
				# initializer = tf.contrib.layers.xavier_initializer()
				initialzier = tf.orthogonal_initializer()
			input_dim = input_dim // 4
			self.Wh = tf.get_variable("Wh", [input_dim, output_dim],
									initializer=initializer)
			self.Wx = tf.get_variable("Wx", [input_dim, output_dim],
									initializer=initializer)
			self.Wy = tf.get_variable("Wy", [input_dim, output_dim],
									initializer=initializer)
			self.Wh = make_quarternion_mul(self.Wh)
			self.Wx = make_quarternion_mul(self.Wx)
			self.Wy = make_quarternion_mul(self.Wy)

	@property
	def state_size(self):
		return self.dim

	@property
	def output_size(self):
		return self.dim


	def __call__(self, inputs, state, scope=None):
		"""
		inputs: 2-D tensor of shape [batch_size, feats + [gates]]
		"""
		new_state = tf.matmul(state, self.Wh) + tf.matmul(inputs, self.Wx)
		new_state = tf.nn.sigmoid(new_state)
		output = tf.nn.tanh(tf.matmul(inputs, self.Wy))
		return output, new_state

#
# def q_xavier_initializer(uniform=True, seed=None, dtype=dtypes.float32):
#   """Returns an initializer performing "Xavier" initialization for weights.
#
#   This function implements the weight initialization from:
#
#   Xavier Glorot and Yoshua Bengio (2010):
# 		   [Understanding the difficulty of training deep feedforward neural
# 		   networks. International conference on artificial intelligence and
# 		   statistics.](
# 		   http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
#
#   This initializer is designed to keep the scale of the gradients roughly the
#   same in all layers. In uniform distribution this ends up being the range:
#   `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
#   deviation of `sqrt(2. / (in + out))` is used.
#
#   Args:
# 	uniform: Whether to use uniform or normal distributed random initialization.
# 	seed: A Python integer. Used to create random seeds. See
# 		  `tf.set_random_seed` for behavior.
# 	dtype: The data type. Only floating point types are supported.
#
#   Returns:
# 	An initializer for a weight matrix.
#   """
#   return q_variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
# 									  uniform=uniform, seed=seed, dtype=dtype)
#
# # xavier_initializer_conv2d = xavier_initializer
#
#
# def q_variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,
# 								 seed=None, dtype=dtypes.float32):
#   """Returns an initializer that generates tensors without scaling variance.
#
#   When initializing a deep network, it is in principle advantageous to keep
#   the scale of the input variance constant, so it does not explode or diminish
#   by reaching the final layer. This initializer use the following formula:
#
#   ```python
# 	if mode='FAN_IN': # Count only number of input connections.
# 	  n = fan_in
# 	elif mode='FAN_OUT': # Count only number of output connections.
# 	  n = fan_out
# 	elif mode='FAN_AVG': # Average number of inputs and output connections.
# 	  n = (fan_in + fan_out)/2.0
#
# 	  truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
#   ```
#
#   * To get [Delving Deep into Rectifiers](
# 	 http://arxiv.org/pdf/1502.01852v1.pdf) (also know as the "MSRA
# 	 initialization"), use (Default):<br/>
# 	`factor=2.0 mode='FAN_IN' uniform=False`
#   * To get [Convolutional Architecture for Fast Feature Embedding](
# 	 http://arxiv.org/abs/1408.5093), use:<br/>
# 	`factor=1.0 mode='FAN_IN' uniform=True`
#   * To get [Understanding the difficulty of training deep feedforward neural
# 	networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
# 	use:<br/>
# 	`factor=1.0 mode='FAN_AVG' uniform=True.`
#   * To get `xavier_initializer` use either:<br/>
# 	`factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
# 	`factor=1.0 mode='FAN_AVG' uniform=False`.
#
#   Args:
# 	factor: Float.  A multiplicative factor.
# 	mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
# 	uniform: Whether to use uniform or normal distributed random initialization.
# 	seed: A Python integer. Used to create random seeds. See
# 		  `tf.set_random_seed` for behavior.
# 	dtype: The data type. Only floating point types are supported.
#
#   Returns:
# 	An initializer that generates tensors with unit variance.
#
#   Raises:
# 	ValueError: if `dtype` is not a floating point type.
# 	TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
#   """
#   if not dtype.is_floating:
# 	raise TypeError('Cannot create initializer for non-floating point type.')
#   if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
# 	raise TypeError('Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)
#
#   # pylint: disable=unused-argument
#   def _initializer(shape, dtype=dtype, partition_info=None):
# 	"""Initializer function."""
# 	if not dtype.is_floating:
# 	  raise TypeError('Cannot create initializer for non-floating point type.')
# 	# Estimating fan_in and fan_out is not possible to do perfectly, but we try.
# 	# This is the right thing for matrix multiply and convolutions.
#
# 	fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
# 	fan_out = float(shape[-1])
# 	s = 1. / np.sqrt(2*(fan_in + fan_out))
# 	#Generating randoms and purely imaginary lights :
#
# 	shape[-1] = shape[-1] // 4
# 	number_of_weights = np.prod(shape)
# 	v_i = np.random.uniform(0.0,1.0,number_of_weights)
# 	v_j = np.random.uniform(0.0,1.0,number_of_weights)
# 	v_k = np.random.uniform(0.0,1.0,number_of_weights)
# 	#Make these purely imaginary lights unitary
# 	for i in range(0, number_of_weights):
# 		norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
# 		v_i[i]/= norm
# 		v_j[i]/= norm
# 		v_k[i]/= norm
# 	v_i = v_i.reshape(shape)
# 	v_j = v_j.reshape(shape)
# 	v_k = v_k.reshape(shape)
#
# 	rng = RandomState(1337)
#
# 	modulus = rng.rayleigh(scale=s, size=shape)
# 	phase = rng.uniform(low=-np.pi, high=np.pi, size=shape)
#
# 	weight_r = modulus * np.cos(phase)
# 	weight_i = modulus * v_i*np.sin(phase)
# 	weight_j = modulus * v_j*np.sin(phase)
# 	weight_k = modulus * v_k*np.sin(phase)
#
# 	weight = np.concatenate([weight_r, weight_i, weight_j, weight_k], axis=-1)
# 	return weight
# 	# if shape:
# 	#   fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
# 	#   fan_out = float(shape[-1])
# 	# else:
# 	#   fan_in = 1.0
# 	#   fan_out = 1.0
# 	# for dim in shape[:-2]:
# 	#   fan_in *= float(dim)
# 	#   fan_out *= float(dim)
# 	# if mode == 'FAN_IN':
# 	#   # Count only number of input connections.
# 	#   n = fan_in
# 	# elif mode == 'FAN_OUT':
# 	#   # Count only number of output connections.
# 	#   n = fan_out
# 	# elif mode == 'FAN_AVG':
# 	#   # Average number of inputs and output connections.
# 	#   n = (fan_in + fan_out) / 2.0
# 	# if uniform:
# 	#   # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
# 	#   limit = math.sqrt(3.0 * factor / n)
# 	#   return random_ops.random_uniform(shape, -limit, limit,
# 	#                                    dtype, seed=seed)
# 	# else:
# 	#   # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
# 	#   trunc_stddev = math.sqrt(1.3 * factor / n)
# 	#   return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
# 	#                                      seed=seed)
#   # pylint: enable=unused-argument
#
#   return _initializer
