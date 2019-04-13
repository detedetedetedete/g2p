import numpy as np
import re
import time
import json
from Seq2Seq import Seq2Seq
from keras import backend as K


def split_record(rec):
  parts = re.split(' {3,}', rec)
  return parts[0], parts[1].split(' ')


def load_model(path):
  with open(path) as model_def_file:
    return json.load(model_def_file)


def load_records():
  return dict(
    [split_record(record) for record in filter(None, open('./g2p.dict', 'r', encoding='utf-8').read().split('\n'))]
  )


def train(model_def_path, epochs=300, batch_size=64):
  model_def = load_model(model_def_path)
  records = load_records()
  model = Seq2Seq(model_def, working_dir=time.strftime(f"./models/{model_def['name']}-%Y_%m_%d-%H%M%S"))
  model.save_no_train()
  model.train(data=records, epochs=epochs, batch_size=batch_size)
  model.save()
  return model


def default():
  model = train("S2S-LSTM152D02R02-LSTM76D02R02-D03-Dense76ReLU-DenseSoftmax----BEST.json", 300, 64)

  while True:
      print('-')
      inp = input("Enter word to decode: ")
      try:
        outp = model.infer(inp)
        print('Decoded graphemes:', outp)
      except BaseException as e:
        print(e.__doc__)
        print(e)


np.set_printoptions(linewidth=np.nan)
if __name__ == '__main__':
  #default()
  models = [
    'nextModels/S2S-GRU256-DenseSoftmax-Nadam.json',
    'nextModels/S2S-LSTM256-DenseSoftmax-Nadam.json',
    'nextModels/S2S-GRU76-GRU152-DenseSoftmax-Nadam.json',
    'nextModels/S2S-LSTM76-LSTM152-DenseSoftmax-Nadam.json',
    'nextModels/S2S-GRU114-GRU38-DenseSoftmax-Nadam.json',
    'nextModels/S2S-LSTM114-LSTM38-DenseSoftmax-Nadam.json',
    'nextModels/S2S-GRU152-GRU76-DenseSoftmax-Nadam.json',
    'nextModels/S2S-LSTM152-LSTM76-DenseSoftmax-Nadam.json',
    'nextModels/S2S-GRU114-Dense38Relu-GRU38-DenseSoftmax-Nadam.json',
    'nextModels/S2S-LSTM114-Dense38Relu-LSTM38-DenseSoftmax-Nadam.json',
    'nextModels/S2S-GRU152-GRU76-Dense76Relu-DenseSoftmax-Nadam.json',
    'nextModels/S2S-LSTM152-LSTM76-Dense76Relu-DenseSoftmax-Nadam.json'
  ]

  for model in models:
    for i in range(0, 1):
      mdl = train(model, 300, 64)
      K.clear_session()
      Seq2Seq(load=True, working_dir=mdl.working_dir)
      K.clear_session()

