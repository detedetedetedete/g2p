from keras import optimizers


class OptimizerResolver(object):
  def __init__(self):
    self.providers = {}

  def add(self, name, provider):
    if name in self.providers:
      print("Warning: Provider '{}' was already defined, will redefine.".format(name))
    self.providers[name] = provider

  def __call__(self, conf, model):
    return self._fallback(conf, model) if conf["type"] not in self.providers else self.providers[conf["type"]](conf, model)

  @staticmethod
  def _fallback(conf, model):
    optimizer_ctor = optimizers.__dict__[conf["type"]]
    if optimizers is None:
      return None
    return optimizer_ctor(*conf["args"], **conf["params"])

