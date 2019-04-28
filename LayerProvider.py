import keras
from keras import layers


class LayerProvider(object):
  def __init__(self):
    self.providers = {}

  def add(self, name, provider):
    if name in self.providers:
      print("Warning: Provider '{}' was already defined, will redefine.".format(name))
    self.providers[name] = provider

  def __getitem__(self, item):
    return self._fallback if item not in self.providers else self.providers[item]

  @staticmethod
  def _fallback(layer, idx, model):
    return keras.__dict__[layer["type"]] if layer["type"] in keras.__dict__ else layers.__dict__[layer["type"]]

