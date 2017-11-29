import numpy as np
from mutual_information_calc import *

#######################################
# GET INFORMATION FROM NEURAL NETWORK #
#######################################
# based on source code https://github.com/ravidziv/IDNNs


# set global variables to discretize outputs of layers
bins = np.linspace(-1, 1, 30)
interval_information_display = 30


# give probabilities to discretized values of the outputs of layers
def extract_probs(label, x):
	"""calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
	pys = np.sum(label, axis=0) / float(label.shape[0])
	b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
	unique_array, unique_indices, unique_inverse_x, unique_counts = \
		np.unique(b, return_index=True, return_inverse=True, return_counts=True)
	unique_a = x[unique_indices]
	b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
	pxs = unique_counts / float(np.sum(unique_counts))
	p_y_given_x = []
	for i in range(0, len(unique_array)):
		indexs = unique_inverse_x == i
		py_x_current = np.mean(label[indexs, :], axis=0)
		p_y_given_x.append(py_x_current)
	p_y_given_x = np.array(p_y_given_x).T
	b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
	unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
		np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
	pys1 = unique_counts_y / float(np.sum(unique_counts_y))
	return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs


# calculates mutual information based on the sampled discretized values
def calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y):
	bins = bins.astype(np.float32)
	num_of_bins = bins.shape[0]

	digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_ts = unique_counts / float(sum(unique_counts))
	PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
	local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
	                                                 unique_array)
	params = {}
	params['local_IXT'] = local_IXT
	params['local_ITY'] = local_ITY
	return params


# calculates mutual information after a specific number of epochs
def calc_information_for_epoch(iter_index, ws_iter_index, unique_inverse_x,
                               unique_inverse_y, label, b, b1,
                               len_unique_a, pys, pxs, py_x, pys1):
	params = np.array(
			[calc_information_sampling(ws_iter_index[i], bins, pys1, pxs, label, b, b1,
			                                                 len_unique_a, py_x, unique_inverse_x,
			                                                 unique_inverse_y)
                                                   for i in range(len(ws_iter_index))])

	return params


# main method to call to collect mutual information values from layers
def get_information(ws, x, label, epoch_num=-1):
	# parameter epoch_num represents at what stage we want to collect the mutual information
	# if epoch_num = -1, we collect mutual information from the evolution of epochs
    pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, x)
    if epoch_num == -1:
        params = np.array([calc_information_for_epoch
	                        (i, ws[i], unique_inverse_x, unique_inverse_y, label,
	                        b, b1, len(unique_a), pys, pxs, p_y_given_x, pys1)
	                        for i in range(len(ws))])
    else:
        params = calc_information_for_epoch(epoch_num-1, ws[epoch_num-1], unique_inverse_x, unique_inverse_y, label,
	                        b, b1, len(unique_a), pys, pxs, p_y_given_x, pys1)
    return params


# extract mutual information values from numpy.array cleanly
def extract_array(data, name):
    results = [[data[j,][name]] for j in range(data.shape[0])]
    return results
