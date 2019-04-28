import numpy as np
import time
from Seq2Seq import Seq2Seq
from keras import backend as K

from utils import load_model, load_records


def train(model_def_path, epochs=300, batch_size=64):
  model_def = load_model(model_def_path)
  records = load_records()
  model = Seq2Seq(model_def, working_dir=time.strftime("./models/{}-%Y_%m_%d-%H%M%S".format(model_def['name'])))
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
    './model_def_phase1_topology/S2S-T1-GRU-Nadam.json',
    './model_def_phase1_topology/S2S-T1-LSTM-Nadam.json',
    './model_def_phase1_topology/S2S-T2-GRU-Nadam.json',
    './model_def_phase1_topology/S2S-T2-LSTM-Nadam.json',
    './model_def_phase1_topology/S2S-T3-GRU-Nadam.json',
    './model_def_phase1_topology/S2S-T3-LSTM-Nadam.json',
    './model_def_phase1_topology/S2S-T4-GRU-Nadam.json',
    './model_def_phase1_topology/S2S-T4-LSTM-Nadam.json',
    './model_def_phase1_topology/S2S-T5-GRU-Nadam-ReLU.json',
    './model_def_phase1_topology/S2S-T5-LSTM-Nadam-ReLU.json',
    './model_def_phase1_topology/S2S-T6-GRU-Nadam-ReLU.json',
    './model_def_phase1_topology/S2S-T6-LSTM-Nadam-ReLU.json',
    './model_def_phase2_activation/S2S-T5-GRU-Nadam-LReLU.json',
    './model_def_phase2_activation/S2S-T5-GRU-Nadam-Sigmoid.json',
    './model_def_phase2_activation/S2S-T5-GRU-Nadam-Tanh.json',
    './model_def_phase2_activation/S2S-T5-GRU-Nadam-Elu.json',
    './model_def_phase2_activation/S2S-T5-LSTM-Nadam-LReLU.json',
    './model_def_phase2_activation/S2S-T5-LSTM-Nadam-Sigmoid.json',
    './model_def_phase2_activation/S2S-T5-LSTM-Nadam-Tanh.json',
    './model_def_phase2_activation/S2S-T5-LSTM-Nadam-Elu.json',
    './model_def_phase2_activation/S2S-T6-GRU-Nadam-LReLU.json',
    './model_def_phase2_activation/S2S-T6-GRU-Nadam-Sigmoid.json',
    './model_def_phase2_activation/S2S-T6-GRU-Nadam-Tanh.json',
    './model_def_phase2_activation/S2S-T6-GRU-Nadam-Elu.json',
    './model_def_phase2_activation/S2S-T6-LSTM-Nadam-LReLU.json',
    './model_def_phase2_activation/S2S-T6-LSTM-Nadam-Sigmoid.json',
    './model_def_phase2_activation/S2S-T6-LSTM-Nadam-Tanh.json',
    './model_def_phase2_activation/S2S-T6-LSTM-Nadam-Elu.json'
  ]

  for model in models:
    for i in range(0, 2):
      mdl = train(model, 300, 64)
      K.clear_session()
      Seq2Seq(load=True, working_dir=mdl.working_dir)
      K.clear_session()

