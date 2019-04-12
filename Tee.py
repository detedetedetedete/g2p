import sys


class Tee(object):
  def __init__(self, name, mode):
    self.file = open(name, mode)
    self.stdout = sys.stdout
    sys.stdout = self

  def close(self):
    sys.stdout = self.stdout
    self.file.close()

  def __enter__(self):
    pass

  def __exit__(self, _type, _value, _traceback):
    self.close()

  def write(self, data):
    self.file.write(data)
    self.stdout.write(data)

  def flush(self):
    self.file.flush()
