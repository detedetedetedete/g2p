import numpy as np


class IOMap(object):
  def __init__(self, el_list):
    self.map = {}
    self.keys = []

    for (i, el) in enumerate(el_list):
      self.map[el] = np.zeros(len(el_list), dtype='float32')
      self.map[el][i] = 1.
      self.keys.append(el)

  def encode(self, el_list, length=None):
    if length is None:
      length = len(el_list)
    result = np.zeros((length, len(self.map)), dtype='float32')
    for(i, el) in enumerate(el_list):
      result[i] = self.map[el]
    return result

  def decode(self, el_list):
    result = []
    for el in el_list:
      if np.max(el) == 0.:
        continue
      result.append(self.keys[np.argmax(el)])
    return result

  def length(self):
    return len(self.keys)
