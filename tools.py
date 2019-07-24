import numpy as np
from scipy.ndimage.interpolation import rotate

class DataLoader:
  def __init__(self,
               name,
               number_of_classes,
               number_of_transformations,
               loaded_size,
               desired_size,
               max_size=None,
               noise=True):

    loaded = np.loadtxt(name)
    if max_size is not None:
      subset = np.random.choice(loaded.shape[0], max_size, replace=False)
      loaded = loaded[subset, :]

    padded_x = self._pad(loaded[:, :-1], loaded_size, desired_size)

    if noise:
          padded_x = padded_x + np.random.uniform(low=0,high=1,size=np.shape(padded_x))

    self._x = self._transform(padded_x, number_of_transformations)
    self._y = self._int_labels_to_one_hot(loaded[:, -1], number_of_classes)
    self._completed_epochs = -1
    self._new_epoch = False
    self._start_new_epoch()

  def _pad(self, loaded_x, loaded_size, desired_size):
    padding_size = (desired_size - loaded_size) // 2
    padding_list = [[0, 0],
                    [padding_size, padding_size],
                    [padding_size, padding_size],
                    [0, 0]]

    return np.pad(np.reshape(loaded_x, [-1, loaded_size, loaded_size, 1]),
                  padding_list,
                  'constant',
                  constant_values=0)

  def _transform(self, padded, number_of_transformations):
    tiled = np.tile(np.expand_dims(padded, 4), [number_of_transformations])
    for transformation_index in range(number_of_transformations):
      angle = 360.0 * transformation_index / float(number_of_transformations)
      tiled[:, :, :, :, transformation_index] = rotate(
          tiled[:, :, :, :, transformation_index],
          angle,
          axes=[1, 2],
          reshape=False)
    print('finished transforming')
    return tiled

  def _int_labels_to_one_hot(self, int_labels, number_of_classes):
    offsets = np.arange(self._size()) * number_of_classes
    one_hot_labels = np.zeros((self._size(), number_of_classes))
    flat_iterator = one_hot_labels.flat
    for index in range(self._size()):
      flat_iterator[offsets[index] + int(int_labels[index])] = 1
    return one_hot_labels

  def _size(self):
    return self._x.shape[0]

  def _start_new_epoch(self):
    permuted_indexes = np.random.permutation(self._size())
    print(np.shape(permuted_indexes))
    print(np.shape(self._x))
    self._x = self._x[permuted_indexes, :]
    self._y = self._y[permuted_indexes]
    self._completed_epochs += 1
    self._index = 0
    self._new_epoch = True

  def get_completed_epochs(self):
    return self._completed_epochs

  def is_new_epoch(self):
    return self._new_epoch

  def next_batch(self, batch_size):
    if (self._new_epoch):
      self._new_epoch = False
    start = self._index
    end = start + batch_size
    if (end > self._size()):
      assert batch_size <= self._size()
      self._start_new_epoch()
      start = 0
      end = start + batch_size
    self._index += batch_size
    return self._x[start:end, :], self._y[start:end]

  def all(self):
    return self._x, self._y
