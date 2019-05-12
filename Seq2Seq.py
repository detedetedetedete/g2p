import json
import os
import time
from itertools import chain
from functools import reduce
import pathlib

import numpy as np
from matplotlib import pyplot
import matplotlib

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
from keras import Input, Model
from keras import backend as K
import tensorflow as tf

from IOMap import IOMap
from LayerProvider import LayerProvider
from OptimizerResolver import OptimizerResolver
from ActivationResolver import ActivationResolver
from Tee import Tee
from copy import deepcopy;


class Seq2Seq(object):
  def __init__(self, model_def=None, load=False, layer_provider=LayerProvider(),
               optimizer_resolver=OptimizerResolver(), working_dir="./", activation_resolver=ActivationResolver()):
    self.layer_provider = layer_provider
    self.optimizer_resolver = optimizer_resolver
    self.activation_resolver = activation_resolver
    self.working_dir = working_dir
    self.training_result = None
    self.custom_objs = {}
    self.history = {"loss": [], "val_loss": []}
    self.last_train_data = {"train": {"in": [], "out": []}, "validate": {"in": [], "out": []}}
    self.rnn_layers = {
      "LSTM": {
        "type": "LSTM",
        "states": [
          "state_h",
          "state_c"
        ],
        "state_count": 0
      },
      "GRU": {
        "type": "GRU",
        "states": [
          "state"
        ],
        "state_count": 0
      }
    }

    for name, layer in self.rnn_layers.items():
      layer["state_count"] = len(layer["states"])

    if load:
      model_def = self._load_model_def()
    else:
      pathlib.Path("{}/checkpoints".format(working_dir)).mkdir(parents=True, exist_ok=True)

    self.start_token = "[S]"
    self.end_token = "[E]"

    self.model_def_original = model_def
    self.model_def = deepcopy(model_def)
    self.preprocess_model_def()

    if "reverse_input" not in self.model_def:
      self.model_def["reverse_input"] = False

    if self.start_token not in self.model_def["out_tokens"]:
      self.model_def["out_tokens"].append(self.start_token)

    if self.end_token not in self.model_def["out_tokens"]:
      self.model_def["out_tokens"].append(self.end_token)

    self.in_map = IOMap(self.model_def["in_tokens"])
    self.out_map = IOMap(self.model_def["out_tokens"])

    if self.model_def["layers"][0]["type"] == "Input":
      self.model_def["layers"].pop(0)

    last_rnn = None
    for i, layer in enumerate(self.model_def["layers"]):
      if "name" not in layer:
        layer["name"] = layer["type"]
      if layer["type"] in self.rnn_layers:
        if last_rnn is not None:
          last_rnn["last_rnn"] = False
        layer["last_rnn"] = True
        last_rnn = layer

    if load:
      self.load_last_data()
      self._rebuild()
    else:
      self._build()

  def _load_model_def(self):
    with open("{}/model.json".format(self.working_dir), "r") as file:
      return json.load(file)

  def preprocess_model_def(self):
    self.model_def["compile"]["optimizer"] = self.optimizer_resolver(self.model_def["compile"]["optimizer"], self)
    for layer in self.model_def["layers"]:
      if "activation" in layer["params"]:
        layer["params"]["activation"] = self.activation_resolver(layer["params"]["activation"], self)
        self.custom_objs[layer["params"]["activation"].__name__] = layer["params"]["activation"]

  @staticmethod
  def _find_layer_by_name(model, name):
    for layer in model.layers:
      if layer.name == name:
        return layer
    return None

  def _rebuild(self):
    print("Loading model...")
    self.training_model = load_model("{}/model.h5".format(self.working_dir), self.custom_objs)

    encoder_in = self._find_layer_by_name(self.training_model, "encoder_input").output
    encoder_out = []

    decoder_in = [self._find_layer_by_name(self.training_model, "decoder_input").output]
    decoder_out = []

    last_decoder_out = decoder_in[0]
    encoder_finished = False
    for i, layer_def in enumerate(self.model_def["layers"]):
      print("Rebuilding {}-{}...".format(layer_def['name'], i))
      if layer_def["type"] in self.rnn_layers:
        for state in self.rnn_layers[layer_def["type"]]["states"]:
          decoder_in.append(
            Input(shape=(layer_def['params']['units'],), name="decoder_{}-{}_{}_input".format(layer_def['name'], i, state))
          )

      if not encoder_finished and layer_def["type"] in self.rnn_layers:
        if layer_def["last_rnn"]:
          encoder_finished = True
        elayer = self._find_layer_by_name(self.training_model, "encoder_{}-{}".format(layer_def['name'], i))
        encoder_out += elayer.output[1:]

      dlayer = self._find_layer_by_name(self.training_model, "decoder_{}-{}".format(layer_def['name'], i))
      if layer_def["type"] in self.rnn_layers:
        dout = dlayer(last_decoder_out, initial_state=decoder_in[-self.rnn_layers[layer_def["type"]]["state_count"]:])
        last_decoder_out = dout[0]
        decoder_out += dout[1:]
      else:
        last_decoder_out = dlayer(last_decoder_out)

    self.encoder_model = Model(inputs=encoder_in, outputs=encoder_out)
    self.decoder_model = Model(inputs=decoder_in, outputs=[last_decoder_out] + decoder_out)
    print("Done.")

  def _build(self):
    print("Building models...")
    encoder_in = Input(shape=(None, self.in_map.length()), name="encoder_input")
    encoder_out = []

    decoder_in = [Input(shape=(None, self.out_map.length()), name="decoder_input")]
    decoder_out = []

    training_in = [encoder_in] + decoder_in

    last_encoder_out = encoder_in
    last_tdecoder_out = decoder_in[0]
    last_decoder_out = decoder_in[0]
    encoder_finished = False
    for i, layer_def in enumerate(self.model_def["layers"]):
      print("Adding {}-{}...".format(layer_def['name'], i))
      layer_ctor = self.layer_provider[layer_def["type"]](layer_def, i, self)
      if layer_def["type"] in self.rnn_layers:
        layer_def["params"]["return_state"] = True
        layer_def["params"]["return_sequences"] = True
        for state in self.rnn_layers[layer_def["type"]]["states"]:
          decoder_in.append(
            Input(shape=(layer_def['params']['units'],), name="decoder_{}-{}_{}_input".format(layer_def['name'], i, state))
          )

      if not encoder_finished:
        eparam = deepcopy(layer_def["params"])
        if i == len(self.model_def["layers"]) - 1:
          eparam["units"] = self.in_map.length()
        if layer_def["type"] in self.rnn_layers and layer_def["last_rnn"]:
          eparam["return_sequences"] = False
          encoder_finished = True
        elayer = layer_ctor(name="encoder_{}-{}".format(layer_def['name'], i), **eparam)
        eout = elayer(last_encoder_out)
        if layer_def["type"] in self.rnn_layers:
          last_encoder_out = eout[0]
          encoder_out += eout[1:]
        else:
          last_encoder_out = eout

      dparam = deepcopy(layer_def["params"])
      if i == len(self.model_def["layers"]) - 1:
        dparam["units"] = self.out_map.length()
      dlayer = layer_ctor(name="decoder_{}-{}".format(layer_def['name'], i), **dparam)
      if layer_def["type"] in self.rnn_layers:
        tdout = dlayer(last_tdecoder_out, initial_state=encoder_out[-self.rnn_layers[layer_def["type"]]["state_count"]:])
        last_tdecoder_out = tdout[0]
        dout = dlayer(last_decoder_out, initial_state=decoder_in[-self.rnn_layers[layer_def["type"]]["state_count"]:])
        last_decoder_out = dout[0]
        decoder_out += dout[1:]
      else:
        last_tdecoder_out = dlayer(last_tdecoder_out)
        last_decoder_out = dlayer(last_decoder_out)

    self.training_model = Model(inputs=training_in, outputs=last_tdecoder_out)
    self.encoder_model = Model(inputs=encoder_in, outputs=encoder_out)
    self.decoder_model = Model(inputs=decoder_in, outputs=[last_decoder_out] + decoder_out)

    print("Compiling the training model...")
    self.training_model.compile(**self.model_def["compile"])
    print("Done.")

  def proccess_training_data(self, training_data, validation_data):
    for record in chain(training_data.items(), validation_data.items()):
      record[1].insert(0, self.start_token)
      record[1].append(self.end_token)

    train_encoder_input, train_decoder_input, train_decoder_output = \
      self.vectorize_input(training_data, self.model_def["max_in_length"], self.model_def["max_out_length"])
    validate_encoder_input, validate_decoder_input, validate_decoder_output = \
      self.vectorize_input(validation_data, self.model_def["max_in_length"], self.model_def["max_out_length"])

    self.last_train_data = {
      "train": {
        "in": train_encoder_input,
        "out": train_decoder_input
      },
      "validate": {
        "in": validate_encoder_input,
        "out": validate_decoder_input
      }
    }

    return train_encoder_input, train_decoder_input, train_decoder_output,\
        validate_encoder_input, validate_decoder_input, validate_decoder_output

  def load_last_data(self):
    files = os.listdir(self.working_dir)
    training_data = None
    validation_data = None
    for file in files:
      if file.startswith("training_data_"):
        with open("{}/{}".format(self.working_dir, file)) as t:
          training_data = json.load(t)
      elif file.startswith("validation_data_"):
        with open("{}/{}".format(self.working_dir, file)) as v:
          validation_data = json.load(v)
      if training_data is not None and validation_data is not None:
        break

    self.proccess_training_data(training_data, validation_data)

  def train(self, data=None, training_data=None, validation_data=None, validation_split=0.3, **kwargs):
    if data is not None:
      train_n = int(len(data) * (1. - validation_split))
      training_words = np.random.choice(list(data.keys()), train_n, False)
      training_data = {k: v for (k, v) in data.items() if k in training_words}
      validation_data = {k: v for (k, v) in data.items() if k not in training_words}

    train_date = time.strftime("%d_%m_%Y-%H%M%S")
    with open("{}/training_data_{}.json".format(self.working_dir, train_date), "w") as file:
      json.dump(training_data, file, indent=2)
    with open("{}/validation_data_{}.json".format(self.working_dir, train_date), "w") as file:
      json.dump(validation_data, file, indent=2)

    train_encoder_input, train_decoder_input, train_decoder_output, \
        validate_encoder_input, validate_decoder_input, validate_decoder_output = \
            self.proccess_training_data(training_data, validation_data)

    tensorboard_callback = TensorBoard(log_dir="{}/tensorboard".format(self.working_dir))
    model_checkpoint = ModelCheckpoint(
      filepath="{}/checkpoints/weights.{{epoch:03d}}-loss{{loss:.4f}}-val_loss{{val_loss:.4f}}.hdf5".format(self.working_dir),
      monitor="val_loss",
      verbose=1,
      save_weights_only=True,
      period=10
    )
    default_callbacks = [tensorboard_callback, model_checkpoint]
    if "callbacks" in kwargs:
      kwargs["callbacks"].extend(default_callbacks)
    else:
      kwargs["callbacks"] = default_callbacks

    kwargs["validation_data"] = ([validate_encoder_input, validate_decoder_input], validate_decoder_output)
    self.training_result = self.training_model.fit([train_encoder_input, train_decoder_input], train_decoder_output, **kwargs)
    self.history = self.training_result.history

  def vectorize_input(self, data, max_enc_length, max_dec_length):
    encoder_input = np.zeros((len(data), max_enc_length, self.in_map.length()), dtype='float32')
    decoder_input = np.zeros((len(data), max_dec_length, self.out_map.length()), dtype='float32')
    decoder_output = np.zeros((len(data), max_dec_length, self.out_map.length()), dtype='float32')

    for i, (key, value) in enumerate(data.items()):
      e_input = key[::-1] if self.model_def["reverse_input"] else key
      encoder_input[i] = self.in_map.encode(e_input, max_enc_length)
      decoder_input[i] = self.out_map.encode(value, max_dec_length)
      decoder_output[i] = self.out_map.encode(value[1:], max_dec_length)

    return encoder_input, decoder_input, decoder_output

  def __infer(self, input, max_length=255):
    states = self.encoder_model.predict(input)
    if len(self.encoder_model.outputs) == 1:
      states = [states]
    result = self.out_map.encode([self.start_token])

    end_frame = self.out_map.encode([self.end_token])[0]
    while np.argmax(result[-1]) != np.argmax(end_frame) and len(result) <= max_length:
      states.insert(0, result[-1].reshape((1, 1,) + result[-1].shape))
      output = self.decoder_model.predict(states)
      states = output[1:]
      result = np.append(result, output[0][0], axis=0)

    return self.out_map.decode(result)

  def _infer(self, input, max_length=255):
    return self.__infer(input.reshape((1,) + input.shape), max_length)

  def infer(self, input, max_length=255):
    input = self.in_map.encode(input, self.model_def["max_in_length"])
    return self._infer(input, max_length)

  def infer_many(self, input_list, max_length=255):
    result = []
    for input in input_list:
      result.append(self.infer(input, max_length))
    return result

  def save_history(self):
    matplotlib.rcParams['figure.figsize'] = (18, 16)
    matplotlib.rcParams['figure.dpi'] = 180
    pyplot.figure()
    pyplot.plot(self.history['loss'])
    pyplot.plot(self.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig("{}/history.pdf".format(self.working_dir))

  def save_summary(self, width=256):
    with Tee("{}/summary.log".format(self.working_dir), 'w'):
      self.training_model.summary(width)

  @staticmethod
  def _acc_report(data, count):
    print("Out of {} test cases:".format(count))
    for err_n, err_count in sorted(data.items()):
      print("\tHad {} mistakes: {} ({:.2f}%)".format(err_n, err_count, err_count / count * 100))

  @staticmethod
  def _write_report_csv(records, file):
    file.write("input;output;expected;errors;\n")
    for rec in records:
      file.write("{};".format(reduce(lambda a, b: "{}{}".format(a, b), rec['input'])) +
                 "{};".format(reduce(lambda a, b: "{}:{}".format(a, b), rec['output'])) +
                 "{};".format(reduce(lambda a, b: "{}:{}".format(a, b), rec['expected_output'])) +
                 "{};\n".format(rec['errors']))
    file.write('\n')

  def validate(self, inputs, real_outputs):
    pairs = zip(inputs, real_outputs)
    accuracy = {}
    for pair in pairs:
      errors = 0
      output = self.infer(pair[0])

      for symbols in zip(output, pair[1]):
        errors += symbols[0] != symbols[1]
      errors += abs(len(output) - len(pair[1]))

      accuracy[errors] = 1 if errors not in accuracy else accuracy[errors] + 1

    return accuracy

  def _generate_report(self, data_type=None):
    accuracy = {"train": {}, "validate": {}, "full": {}}
    report = {"train": [], "validate": [], "full": []}

    if data_type is None:
      data_type = self.last_train_data

    for type in data_type:
      for idx in range(0, len(self.last_train_data[type]["in"])):
        input = self.last_train_data[type]["in"][idx]
        true_output = self.out_map.decode(self.last_train_data[type]["out"][idx])
        output = self._infer(input)

        errors = 0
        for offset in range(min(len(output), len(true_output))):
          errors += output[offset] != true_output[offset]
        errors += abs(len(output) - len(true_output))

        accuracy[type][errors] = 1 if errors not in accuracy[type] else accuracy[type][errors] + 1
        accuracy["full"][errors] = 1 if errors not in accuracy["full"] else accuracy["full"][errors] + 1
        record = {
          "input": self.in_map.decode(input),
          "output": output,
          "expected_output": true_output,
          "errors": errors
        }
        report[type].append(record)
        report["full"].append(record)

    return accuracy, report

  def save_accuracy(self, accuracy, report, name):
    train = accuracy["train"][0] if 0 in accuracy["train"] else 0
    val = accuracy["validate"][0] if 0 in accuracy["validate"] else 0
    full = accuracy["full"][0] if 0 in accuracy["full"] else 0
    with Tee("{}/{}-{:.2f}-{:.2f}-{:.2f}.log".format(
              self.working_dir,
              name,
              train / (1 if len(report["train"]) == 0 else len(report["train"])) * 100,
              val / (1 if len(report["validate"]) == 0 else len(report["validate"])) * 100,
              full / (1 if len(report["full"]) == 0 else len(report["full"])) * 100
          ),
          'a'
      ):
      print('------------------Accuracy----------------------')
      print('-----------------Train set----------------------')
      self._acc_report(accuracy["train"], len(report["train"]))
      print('---------------Validation set-------------------')
      self._acc_report(accuracy["validate"], len(report["validate"]))
      print('------------------Full set----------------------')
      self._acc_report(accuracy["full"], len(report["full"]))
      print('------------------------------------------------')

  def save_report(self, report, name):
    for rep in report:
      with open("{}/{}_{}.csv".format(self.working_dir, name, rep), "w") as file:
        self._write_report_csv(report[rep], file)

  def save_full_report(self, accuracy_name='accuracy', report_name='result'):
    accuracy, report = self._generate_report()
    self.save_accuracy(accuracy, report, accuracy_name)
    self.save_report(report, report_name)

  def load_weights(self, path):
    self.training_model.load_weights(path)

  def evaluate_checkpoints(self, progress=lambda p: None, data_type=None):
    if not os.path.isdir("{}/checkpoints".format(self.working_dir)):
      print("No checkpoint folder!")
      return
    checkpoints = [file for file in os.listdir("{}/checkpoints".format(self.working_dir))
                   if file.startswith('weights.') and file.endswith('.hdf5')]
    for idx, checkpoint in enumerate(sorted(checkpoints)):
      path = "{}/checkpoints/{}".format(self.working_dir, checkpoint)
      print("Evaluating {}...".format(path))
      self.load_weights(path)
      relative_path = "checkpoints/{}".format(checkpoint)
      accuracy, report = self._generate_report(data_type)
      self.save_accuracy(accuracy, report, relative_path)
      progress(idx+1)
    self.load_weights("{}/model-weights.h5".format(self.working_dir))

  def save_model(self):
    self.training_model.save("{}/model.h5".format(self.working_dir))
    self.training_model.save_weights("{}/model-weights.h5".format(self.working_dir))

  def save_for_inference_tf(self):
    print('Saving tf models for inference...\n\tIdentifying encoder output names...')
    rnn_layers = list(
      map(
        lambda x: self.rnn_layers[x['type']],
        filter(
          lambda x: x['type'] in self.rnn_layers,
          self.model_def['layers']
        )
      )
    )

    tf.identity(self.encoder_model.inputs[0], 'encoder_input')
    encoder_output_names = []

    enc_idx = 0
    for layer_idx, layer in enumerate(rnn_layers):
      for state in layer['states']:
        encoder_output_names.append("encoder_{}-{}_{}_output".format(layer["type"], layer_idx, state))
        tf.identity(self.encoder_model.outputs[enc_idx], encoder_output_names[-1])
        enc_idx += 1
    print("\t{}".format(encoder_output_names))

    print("\tConverting encoder variables to constants...")
    session = K.get_session()
    encoder_model_const = tf.graph_util.convert_variables_to_constants(
      session,
      session.graph.as_graph_def(),
      encoder_output_names)

    print("\tSaving encoder inference model...")
    tf.io.write_graph(encoder_model_const, self.working_dir, "encoder_inference_model.pbtxt", as_text=True)
    tf.io.write_graph(encoder_model_const, self.working_dir, "encoder_inference_model.pb", as_text=False)

    print("\tIdentifying decoder output names...")
    tf.identity(self.decoder_model.inputs[0], 'decoder_input')
    decoder_output_names = ['decoder_output']
    tf.identity(self.decoder_model.outputs[0], decoder_output_names[0])

    dec_idx = 1
    for layer_idx, layer in enumerate(rnn_layers):
      for state in layer['states']:
        decoder_output_names.append("decoder_{}-{}_{}_output".format(layer["type"], layer_idx, state))
        tf.identity(self.decoder_model.outputs[dec_idx], decoder_output_names[-1])
        tf.identity(self.decoder_model.inputs[dec_idx], "decoder_{}-{}_{}_input".format(layer["type"], layer_idx, state))
        dec_idx += 1

    print("\t{}".format(decoder_output_names))

    print("\tConverting decoder variables to constants...")
    decoder_model_const = tf.graph_util.convert_variables_to_constants(
      session,
      session.graph.as_graph_def(),
      decoder_output_names
    )

    print("\tSaving decoder inference model...")
    tf.io.write_graph(decoder_model_const, self.working_dir, "decoder_inference_model.pbtxt", as_text=True)
    tf.io.write_graph(decoder_model_const, self.working_dir, "decoder_inference_model.pb", as_text=False)
    print("Saving tf models for inference - Done")

  def save_model_plots(self):
    plot_model(self.training_model, to_file="{}/model.png".format(self.working_dir), show_layer_names=True, show_shapes=True)
    plot_model(self.encoder_model, to_file="{}/encoder_model.png".format(self.working_dir), show_layer_names=True, show_shapes=True)
    plot_model(self.decoder_model, to_file="{}/decoder_model.png".format(self.working_dir), show_layer_names=True, show_shapes=True)

  def save_no_train(self, width=256):
    #self.save_model_plots()
    self.save_summary(width=width)

  def save_model_def(self):
    with open("{}/model.json".format(self.working_dir), "w") as file:
      self.model_def["out_tokens"].remove(self.start_token)
      self.model_def["out_tokens"].remove(self.end_token)

      json.dump(self.model_def_original, file, indent=2)

      self.model_def["out_tokens"].append(self.start_token)
      self.model_def["out_tokens"].append(self.end_token)

  def save(self, width=256):
    self.save_model()
    self.save_model_def()
    self.save_no_train(width)
    self.save_history()
    self.save_full_report()
    self.save_for_inference_tf()



'''
  Add a class Seq2Seq model
  Builder should set all needed fields, so that
  the model would be training and inference capable
'''

'''
  Training model:
    in: encoder inputs (Input layer), decoder inputs (Input layer)
    out: decoder outputs
  Encoder model:
    in: encoder inputs (Input layer)
    out: all encoder LSTM layer states
  Decoder model:
    in: decoder inputs (Input layer), state inputs for all LSTM layers
    out: decoder outputs, all LSTM layer states
'''

