import numpy as np
from Seq2Seq import Seq2Seq

np.set_printoptions(linewidth=np.nan)


def process_csv(csv):
  csv.pop(0)
  input = []
  output = []
  for entry in csv:
    parts = entry.split(";")
    if len(parts) < 4:
      continue
    input.append(parts[0])
    output.append(parts[2].split(":"))
  return input, output


def validate_model(path):
  with open("{}/result_train.csv".format(path), "r") as ftrain:
    train = process_csv(ftrain.readlines())
  with open("{}/result_validate.csv".format(path), "r") as fvalidate:
    validate = process_csv(fvalidate.readlines())
  full = [train[0]+validate[0], train[1]+validate[1]]

  model = Seq2Seq(load=True, working_dir=path)
  # model.save_for_inference_tf()

  train_acc = model.validate(train[0], train[1])
  model._acc_report(train_acc, len(train[0]))

  validate_acc = model.validate(validate[0], validate[1])
  model._acc_report(validate_acc, len(validate[0]))

  full_acc = model.validate(full[0], full[1])
  model._acc_report(full_acc, len(full[0]))

  return model


model = validate_model("models/model-07_04_2019-142244-96.08-98.52")

while True:
    print('-')
    inp = input("Enter word to decode: ")
    try:
      outp = model.infer(inp)
      print('Decoded graphemes:', outp)
    except BaseException as e:
      print(e.__doc__)
      print(e)
