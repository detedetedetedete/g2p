import numpy as np
import re
import time
import json
from Seq2Seq import Seq2Seq


def split_record(rec):
  parts = re.split(' {3,}', rec)
  return parts[0], parts[1].split(' ')


np.set_printoptions(linewidth=np.nan)

with open("model.json") as model_def_file:
  model_def = json.load(model_def_file)

records = dict(
    [split_record(record) for record in filter(None, open('./g2p.dict', 'r', encoding='utf-8').read().split('\n'))]
)

model = Seq2Seq(model_def, working_dir=time.strftime("./models/model-%d_%m_%Y-%H%M%S"))
model.save_no_train()
model.train(data=records, epochs=600, batch_size=64)
model.save()

while True:
    print('-')
    inp = input("Enter word to decode: ")
    try:
      outp = model.infer(inp)
      print('Decoded graphemes:', outp)
    except BaseException as e:
      print(e.__doc__)
      print(e)
