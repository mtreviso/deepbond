import numpy as np
from numpy.lib.stride_tricks import as_strided


def unroll(list_of_lists, rec=False):
	"""
	:param list_of_lists: a list that contain lists
	:param rec: unroll recursively
	:return: a flattened list
	""" 
	new_list = [item for l in list_of_lists for item in l]
	if rec and isinstance(new_list[0], (np.ndarray, list)):
		return unroll(new_list, rec=rec)
	return new_list

def row_matrix(sequences, force_3d=True, flatten_last_axis=False, dtype=np.float32):
	"""
	:param sequences: a list that may contain n-1 lists
	:param flatten_last_axis: ravel the last two axis
	:return: a np array with shape (1, )
	"""
	tensor = np.array(sequences, dtype=dtype)
	new_shape = (1,) + tensor.shape
	if flatten_last_axis or (force_3d and len(tensor.shape) > 2):
		new_shape = (1, tensor.shape[0], np.prod(tensor.shape[1:]))
	tensor = tensor.reshape(new_shape)
	return tensor

def column_matrix(sequences, force_3d=True, flatten_last_axis=False, dtype=np.float32):
	return row_matrix(sequences, force_3d=force_3d, flatten_last_axis=flatten_last_axis, dtype=dtype).T

def reshape_like(sequences, map_with=None):
	"""
	:param sequences: a list that contains words or ids
	:param map_with: a list of lists that have the same nb of content as :sequences:
	:return: a list of lists with :map_with: structure
	"""
	new_sequences = []
	t = 0
	for seq in map_with:
		new_sequences.append(sequences[t:t+len(seq)])
		t += len(seq)
	return new_sequences


def pad_sequences(sequences, maxlen=None, mask_value=0):
	"""
	:param sequences: list of sequence of ids
	:param maxlen: if not specified, maxlen is max sentence length
	:param mask_value: the value to be used for padding
	:return: a np array with shape (nb_samples, maxlen)
	"""

	dtype = 'int32'
	nb_samples = len(sequences)
	if maxlen is None:
		maxlen = np.max([len(s) for s in sequences])

	x = (np.ones((nb_samples, maxlen)) * mask_value).astype(dtype)
	for idx, s in enumerate(sequences):
		x[idx, :len(s)] = s
	return x


def pad_sequences_3d(sequences, maxlen=None, mask_value=0):
	"""
	:param sequences: list of sequence of ids
	:param maxlen: if not specified, maxlen is max sentence length
	:param mask_value: the value to be used for padding
	:return: a np array with shape (nb_samples, maxlen)
	"""

	dtype = 'int32'
	nb_samples = len(sequences)
	nb_features = len(sequences[0][0])
	if maxlen is None:
		maxlen = np.max([len(s) for s in sequences])

	x = (np.ones((nb_samples, maxlen, nb_features)) * mask_value).astype(dtype)
	for idx, s in enumerate(sequences):
		x[idx, :len(s)] = s
	return x


def unpad_sequences(padded_sequences, map_with=None, mask_value=0):
	"""
	:param padded_sequences: a np array that was padded using pad_sequences
	:param map_with: a list of lists that have the same nb of content as :padded_sequences:
	:param mask_value: the value to be used for padding
	:return: a list with the original sequence structure
	"""

	unpadded = []
	if map_with is not None:
		for i, sequence in enumerate(map_with):
			unpadded.extend(padded_sequences[i, :len(sequence)].tolist())
	else:
		for sequence in padded_sequences:
			unpadded.extend(sequence[sequence != mask_value].tolist())
	return np.array(unpadded)


def vectorize(tensor, one_hot_dim=None):
	"""
	:param tensor: numpy array of sequences of ids
	:param one_hot_dim: if not specified, max value in tensor + 1 is used
	:return:
	"""

	# return np_utils.to_categorical(tensor)
	if not one_hot_dim:
		one_hot_dim = tensor.max() + 1

	if len(tensor.shape) == 1:
		# It's a vector; return a 2d tensor
		tensor_2d = np.zeros((tensor.shape[0], one_hot_dim), dtype=np.bool8)
		for i, val in np.ndenumerate(tensor):
			tensor_2d[i, val] = 1
		return tensor_2d

	tensor_3d = np.zeros((tensor.shape[0], tensor.shape[1], one_hot_dim), dtype=np.bool8)
	for (i, j), val in np.ndenumerate(tensor):
		if val < one_hot_dim:
			tensor_3d[i, j, val] = 1
	return tensor_3d


def unvectorize(tensor):
	"""
	:param tensor: numpy array of sequences of ids
	:return: the row indices that maximizes this tensor
	"""

	return tensor.argmax(axis=-1)


# for 2d
def unconvolve_sequences(window):
	"""
	:param window: a numpy array of sequences of ids that was windowed
	:return: the middle column
	"""

	if len(window.shape) == 1:
		# it is already a vector
		return window
	middle = window.shape[1] // 2
	return window[:, middle]


# for 3d
def unconvolve_sequences_3d(window):
	"""
	:param window: a 1-hot numpy array of sequences of ids that was windowed
	:return: the middle column of the window without unvectorized
	"""

	seq = np.ones((window.shape[0], 1))
	middle = window.shape[1] // 2
	for i, row in enumerate(window):
		seq[i] = unvectorize(row[middle])
	return seq


def convolve_sequences(sequences, window_size, left_pad_value=None, right_pad_value=None):

	"""
	Convolve around each element in each sequence.

	:param sequences: list of lists with possibly varying sizes
	:param window_size: if odd, align elements at the center; if even, align elements at the right  # FIXME: not implemented like that (might not be needed,
	because left and right padding equal to None would produce the same effect)
	:param left_pad_value, right_pad_value: padding values for windows of elements at the start and end of sequences; if equal to None, the corresponding
	padding is not added (see consequence on the return value)
	:return: convolved sequences of size (number of words, window_size) (unless one or both padding values are None, in which case some elements will only
	appear in the context of others, but not aligned by themselves)
	"""
	if left_pad_value is not None:
		left_pad = np.ones(window_size // 2, dtype=np.int) * left_pad_value
	if right_pad_value is not None:
		right_pad = np.ones(window_size // 2, dtype=np.int) * right_pad_value

	def pad_sequence(sequence):
		sequence = np.array(sequence)
		if left_pad_value is not None:
			sequence = np.hstack((left_pad, sequence))
		if right_pad_value is not None:
			sequence = np.hstack((sequence, right_pad))
		return sequence

	lines = sum(len(ws) for ws in sequences)
	window_array = np.zeros((lines, window_size), dtype=np.int)

	i = 0
	for seq in sequences:
		if window_size == 1:
			vectors = pad_sequence(seq).reshape(len(seq), window_size)
		else:
			vectors = ngrams_via_striding(pad_sequence(seq), window_size)
		window_array[i:i + len(vectors), :] = vectors
		i += len(vectors)
	return window_array


def convolve_sequences_3d(sequences, window_size, left_pad_value=None, right_pad_value=None):

	lines = sum(map(len, sequences))
	nb_features = len(sequences[0][0])
	window_array = np.zeros((lines, window_size, nb_features))

	if left_pad_value is not None:
		left_pad = np.ones((window_size // 2, nb_features), dtype=np.int) * left_pad_value
	if right_pad_value is not None:
		right_pad = np.ones((window_size // 2, nb_features), dtype=np.int) * right_pad_value

	def pad_sequence(sequence):
		sequence = np.array(sequence)
		if left_pad_value is not None:
			sequence = np.vstack((left_pad, sequence))
		if right_pad_value is not None:
			sequence = np.vstack((sequence, right_pad))
		return sequence

	i = 0
	for seq in sequences:
		if window_size == 1:
			vectors = pad_sequence(seq).reshape(len(seq), window_size)
		else:
			x = pad_sequence(seq)
			vectors = slide_vectors(x, window_size)
		window_array[i:i + len(vectors), :] = vectors
		i += len(vectors)
	return window_array


def ngrams_via_striding(array, order):
	# https://gist.github.com/mjwillson/060644552eb037ebb3e7
	itemsize = array.itemsize
	assert array.strides == (itemsize,)
	return as_strided(array, (max(array.size + 1 - order, 0), order), (itemsize, itemsize))



def rolling_2d(array, win_h, win_w, step_h, step_w):
	# adapted from: http://www.itsprite.com/pythonefficient-numpy-2d-array-construction-from-1d-array/
	h, w = array.shape
	shape = (((h - win_h) / step_h + 1) * ((w - win_w) / step_w + 1), win_h, win_w)
	strides = (step_w * array.itemsize, win_w * array.itemsize, array.itemsize)
	shape = tuple(map(int, shape))
	strides = tuple(map(int, strides))
	return as_strided(array, shape=shape, strides=strides)


def slide_vectors(array, order):
	return rolling_2d(array, order, array.shape[1], 1, array.shape[1])


def bucketize(data, max_bucket_size=32):
	lengths_counter = {}
	data_by_length = {}
	get_index = lambda l: '%d_%d' % (l, lengths_counter[l]) if l in lengths_counter else '%d_0' % l
	for i, l in enumerate(map(len, data)):
		if get_index(l) not in data_by_length:
			lengths_counter[l] = 0
			data_by_length[get_index(l)] = []
		if len(data_by_length[get_index(l)]) == max_bucket_size:
			lengths_counter[l] += 1
			data_by_length[get_index(l)] = []
		data_by_length[get_index(l)].append(i)
	lengths = list(data_by_length.keys())
	return lengths, data_by_length


def reorder_buckets(preds, golds, lengths, data_by_length):
	new_preds = [None for _ in range(len(preds))]
	new_golds = [None for _ in range(len(golds))]
	i = 0
	for l in lengths:
		for key in data_by_length[l]:
			new_preds[key] = preds[i]
			new_golds[key] = golds[i]
			i += 1
	return new_preds, new_golds
