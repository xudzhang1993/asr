import tensorflow as tf
from scipy import sparse
from tensorflow.python.ops import lookup_ops
import numpy as np


def get_max_time(tensor, time_axis):
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]


def edit_distance(predict, target):
    sparse1 = truncate_to_sparse_tensor(predict)
    sparse2 = sparse.coo_matrix(target)
    sparse1 = tf.SparseTensor(*sparse1)
    sparse2 = tf.SparseTensor(zip(sparse2.row, sparse2.col), sparse2.data, sparse2.shape)
    return tf.edit_distance(sparse1, sparse2, normalize=True)


def truncate_to_sparse_tensor(predict):
    indices = []
    val = []
    shape = predict.shape
    for i, row in enumerate(predict):
        for j, v in enumerate(row):
            if v != '':
                indices.append((i, j))
                val.append(v)
            else:
                break
    return indices, val, shape


def generate_voc_table():
    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', \
                 'v', 'w', 'y', 'z', 'zh', '<SOS>', '']
    return lookup_ops.index_to_string_table_from_tensor(group_phn)


def generate_voc_table_61():
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', \
           'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', \
           'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', \
           'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', \
           'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', \
           'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', \
           'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', \
           'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', \
           'v', 'w', 'y', 'z', 'zh', '<SOS>', '']
    return lookup_ops.index_to_string_table_from_tensor(phn)


def print_variables():
    for v in tf.global_variables():
        print v.name


def truncate(a):
    a = a.tolist()
    try:
        i = a.index('')
    except ValueError:
        i = len(a)
    return np.array(a[:i])


def map_phoneme(target):
    ''' turn 2-D List to SparseTensor
    '''
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', \
           'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', \
           'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', \
           'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', \
           'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', \
           'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', \
           'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', \
           'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', \
           'v', 'w', 'y', 'z', 'zh']

    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', \
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', \
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#', \
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', \
                 'v', 'w', 'y', 'z', 'zh']

    '''
    for phn level, we should collapse 61 labels into 39 labels before scoring

    Reference:
      Heterogeneous Acoustic Measurements and Multiple Classifiers for Speech Recognition(1986), 
        Andrew K. Halberstadt, https://groups.csail.mit.edu/sls/publications/1998/phdthesis-drew.pdf
    '''
    for row in target:
        for i in range(len(row)):
            if row[i] in mapping.keys():
                row[i] = mapping[row[i]]

def create_alignment_image(attention_state):
    alignment = attention_state.alignment_history
    print alignment
    alignment = alignment.stack()
    image = tf.expand_dims(tf.transpose(alignment, [1, 2, 0]), -1)
    return image
if __name__ == "__main__":
    a = np.array([['s', 'g', '', ''],
                  ['a', 'g', 'g', '']])
