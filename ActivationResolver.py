import collections

from keras import activations
import urllib.parse as parse


def enc(strg):
  return parse.quote(str(strg), safe='')


def dec(strg):
  return parse.unquote(strg)

class ActivationResolver(object):
  def __init__(self):
    self.providers = {}

  def add(self, name, provider):
    if name in self.providers:
      print(f"Warning: Provider '{name}' was already defined, will redefine.")
    self.providers[name] = provider

  def __call__(self, conf, model):
    if not isinstance(conf, collections.Mapping):
      conf = {
        "type": conf,
        "params": {}
      }
    return self._fallback(conf, model) if conf["type"] not in self.providers else self.providers[conf["type"]](conf, model)

  @staticmethod
  def _fallback(conf, model):
    activation_fn = activations.__dict__[conf["type"]]
    if activation_fn is None:
      return None
    fn = lambda x: activation_fn(x, **conf["params"])
    name = "activation://{}".format(enc(conf["type"]))
    for idx, (actname, value) in enumerate(conf["params"].items()):
      name += "{}type={}".format("?" if idx == 0 else "&", enc(value.__class__.__name__))
      name += "&name={}".format(enc(actname))
      name += "&value={}".format(enc(value))
    fn.__name__ = name
    return fn