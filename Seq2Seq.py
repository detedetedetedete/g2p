import json
import time
from itertools import chain
from functools import reduce
import pathlib

import numpy as np
from matplotlib import pyplot
import matplotlib

from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.models import load_model
from keras import Input, Model
from keras import backend as K
import tensorflow as tf

from IOMap import IOMap
from LayerProvider import LayerProvider
from OptimizerResolver import OptimizerResolver
from Tee import Tee


class Seq2Seq(object):
  def __init__(self, model_def=None, load=False, layer_provider=LayerProvider(),
               optimizer_resolver=OptimizerResolver(), working_dir="./"):
    self.layer_provider = layer_provider
    self.optimizer_resolver = optimizer_resolver
    self.working_dir = working_dir
    self.training_result = None
    self.history = {"loss": [], "val_loss": []}
    self.last_train_data = {"train": {"in": [], "out": []}, "validate": {"in": [], "out": []}}

    if load:
      model_def = self._load_model_def()
    else:
      pathlib.Path(working_dir).mkdir(parents=True, exist_ok=True)

    self.start_token = "[S]"
    self.end_token = "[E]"

    self.model_def = model_def

    if self.start_token not in self.model_def["out_tokens"]:
      self.model_def["out_tokens"].append(self.start_token)

    if self.end_token not in self.model_def["out_tokens"]:
      self.model_def["out_tokens"].append(self.end_token)

    self.in_map = IOMap(self.model_def["in_tokens"])
    self.out_map = IOMap(self.model_def["out_tokens"])

    if model_def["layers"][0]["type"] == "Input":
      model_def["layers"].pop(0)

    last_lstm = None
    for i, layer in enumerate(model_def["layers"]):
      if "name" not in layer:
        layer["name"] = layer["type"]
      if layer["type"] == "LSTM":
        if last_lstm is not None:
          last_lstm["last_lstm"] = False
        layer["last_lstm"] = True
        last_lstm = layer

    if load:
      self._rebuild()
    else:
      self._build()

  def _load_model_def(self):
    with open(f"{self.working_dir}/model.json", "r") as file:
      return json.load(file)

  @staticmethod
  def _find_layer_by_name(model, name):
    for layer in model.layers:
      if layer.name == name:
        return layer
    return None

  def _rebuild(self):
    print("Loading model...")
    self.training_model = load_model(f"{self.working_dir}/model.h5")

    encoder_in = self._find_layer_by_name(self.training_model, "encoder_input").output
    encoder_out = []

    decoder_in = [self._find_layer_by_name(self.training_model, "decoder_input").output]
    decoder_out = []

    last_decoder_out = decoder_in[0]
    encoder_finished = False
    for i, layer_def in enumerate(self.model_def["layers"]):
      print(f"Rebuilding {layer_def['name']}-{i}...")
      if layer_def["type"] == "LSTM":
        decoder_in += [
          Input(shape=(layer_def['params']['units'],), name=f"decoder_{layer_def['name']}-{i}_state_h_input"),
          Input(shape=(layer_def['params']['units'],), name=f"decoder_{layer_def['name']}-{i}_state_c_input")
        ]

      if not encoder_finished and layer_def["type"] == "LSTM":
        if layer_def["last_lstm"]:
          encoder_finished = True
        elayer = self._find_layer_by_name(self.training_model, f"encoder_{layer_def['name']}-{i}")
        encoder_out += elayer.output[1:]

      dlayer = self._find_layer_by_name(self.training_model, f"decoder_{layer_def['name']}-{i}")
      if layer_def["type"] == "LSTM":
        dout = dlayer(last_decoder_out, initial_state=decoder_in[-2:])
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
      print(f"Adding {layer_def['name']}-{i}...")
      layer_ctor = self.layer_provider[layer_def["type"]](layer_def, i, self)
      if layer_def["type"] == "LSTM":
        layer_def["params"]["return_state"] = True
        layer_def["params"]["return_sequences"] = True

        decoder_in += [
          Input(shape=(layer_def['params']['units'],), name=f"decoder_{layer_def['name']}-{i}_state_h_input"),
          Input(shape=(layer_def['params']['units'],), name=f"decoder_{layer_def['name']}-{i}_state_c_input")
        ]

      if not encoder_finished:
        eparam = layer_def["params"].copy()
        if i == len(self.model_def["layers"]) - 1:
          eparam["units"] = self.in_map.length()
        if layer_def["last_lstm"]:
          eparam["return_sequences"] = False
          encoder_finished = True
        elayer = layer_ctor(name=f"encoder_{layer_def['name']}-{i}", **eparam)
        eout = elayer(last_encoder_out)
        if layer_def["type"] == "LSTM":
          last_encoder_out = eout[0]
          encoder_out += eout[1:]
        else:
          last_encoder_out = eout

      dparam = layer_def["params"].copy()
      if i == len(self.model_def["layers"]) - 1:
        dparam["units"] = self.out_map.length()
      dlayer = layer_ctor(name=f"decoder_{layer_def['name']}-{i}", **dparam)
      if layer_def["type"] == "LSTM":
        tdout = dlayer(last_tdecoder_out, initial_state=encoder_out[-2:])
        last_tdecoder_out = tdout[0]
        dout = dlayer(last_decoder_out, initial_state=decoder_in[-2:])
        last_decoder_out = dout[0]
        decoder_out += dout[1:]
      else:
        last_tdecoder_out = dlayer(last_tdecoder_out)
        last_decoder_out = dlayer(last_decoder_out)

    self.training_model = Model(inputs=training_in, outputs=last_tdecoder_out)
    self.encoder_model = Model(inputs=encoder_in, outputs=encoder_out)
    self.decoder_model = Model(inputs=decoder_in, outputs=[last_decoder_out] + decoder_out)

    print("Compiling the training model...")
    compile_params = self.model_def["compile"].copy()
    compile_params["optimizer"] = self.optimizer_resolver(compile_params["optimizer"], self)
    self.training_model.compile(**compile_params)
    print("Done.")

  def train(self, data=None, training_data=None, validation_data=None, validation_split=0.3, **kwargs):
    if data is not None:
      train_n = int(len(data) * (1. - validation_split))
      training_words = np.random.choice(list(data.keys()), train_n, False)
      training_data = {k: v for (k, v) in data.items() if k in training_words}
      validation_data = {k: v for (k, v) in data.items() if k not in training_words}

    train_date = time.strftime("%d_%m_%Y-%H%M%S")
    with open(f"{self.working_dir}/training_data_{train_date}.json", "w") as file:
      json.dump(training_data, file, indent=2)
    with open(f"{self.working_dir}/validation_data_{train_date}.json", "w") as file:
      json.dump(validation_data, file, indent=2)

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

    tensorboard_callback = TensorBoard(log_dir=f"{self.working_dir}/tensorboard", histogram_freq=10)
    if "callbacks" in kwargs:
      kwargs["callbacks"].append(tensorboard_callback)
    else:
      kwargs["callbacks"] = [tensorboard_callback]

    kwargs["validation_data"] = ([validate_encoder_input, validate_decoder_input], validate_decoder_output)
    self.training_result = self.training_model.fit([train_encoder_input, train_decoder_input], train_decoder_output, **kwargs)
    self.history = self.training_result.history

  def vectorize_input(self, data, max_enc_length, max_dec_length):
    encoder_input = np.zeros((len(data), max_enc_length, self.in_map.length()), dtype='float32')
    decoder_input = np.zeros((len(data), max_dec_length, self.out_map.length()), dtype='float32')
    decoder_output = np.zeros((len(data), max_dec_length, self.out_map.length()), dtype='float32')

    for i, (key, value) in enumerate(data.items()):
      encoder_input[i] = self.in_map.encode(key, max_enc_length)
      decoder_input[i] = self.out_map.encode(value, max_dec_length)
      decoder_output[i] = self.out_map.encode(value[1:], max_dec_length)

    return encoder_input, decoder_input, decoder_output

  def __infer(self, input, max_length=255):
    states = self.encoder_model.predict(input)
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
    pyplot.plot(self.history['loss'])
    pyplot.plot(self.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig(f"{self.working_dir}/history.pdf")

  def save_summary(self, width=256):
    with Tee(f"{self.working_dir}/summary.log", 'w'):
      self.training_model.summary(width)

  @staticmethod
  def _acc_report(data, count):
    print(f'Out of {count} test cases:')
    for err_n, err_count in sorted(data.items()):
      print(f'\tHad {err_n} mistakes: {err_count} ({"{0:.2f}".format(err_count / count * 100)}%)')

  @staticmethod
  def _write_report_csv(records, file):
    file.write("input;output;expected;errors;\n")
    for rec in records:
      file.write(f"{reduce(lambda a,b: f'{a}{b}', rec['input'])};"
                 f"{reduce(lambda a,b: f'{a}:{b}', rec['output'])};"
                 f"{reduce(lambda a,b: f'{a}:{b}', rec['expected_output'])};"
                 f"{rec['errors']};\n")
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

  def _generate_report(self):
    accuracy = {"train": {}, "validate": {}, "full": {}}
    report = {"train": [], "validate": [], "full": []}

    for type in self.last_train_data:
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

  def save_report(self):
    accuracy, report = self._generate_report()

    with Tee(f"{self.working_dir}/summary.log", 'a'):
      print('------------------Accuracy----------------------')
      print('-----------------Train set----------------------')
      self._acc_report(accuracy["train"], len(report["train"]))
      print('---------------Validation set-------------------')
      self._acc_report(accuracy["validate"], len(report["validate"]))
      print('------------------Full set----------------------')
      self._acc_report(accuracy["full"], len(report["full"]))
      print('------------------------------------------------')

    for rep in report:
      with open(f"{self.working_dir}/result_{rep}.csv", "w") as file:
        self._write_report_csv(report[rep], file)

  def save_model(self):
    self.training_model.save(f"{self.working_dir}/model.h5")

  def save_for_inference_tf(self):
    print('Saving tf models for inference...\n\tIdentifying encoder output names...')
    tf.identity(self.encoder_model.inputs[0], 'encoder_input')
    encoder_output_names = []
    for idx in range(0, int(len(self.encoder_model.outputs) / 2)):
      encoder_output_names += [f'encoder_LSTM{idx}_state_h_output', f'encoder_LSTM{idx}_state_c_output']
      tf.identity(self.encoder_model.outputs[idx*2+0], encoder_output_names[-2])
      tf.identity(self.encoder_model.outputs[idx*2+1], encoder_output_names[-1])
    print(f"\t{encoder_output_names}")

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
    for idx in range(0, int(len(self.decoder_model.outputs) / 2)):
      decoder_output_names += [f'decoder_LSTM{idx}_state_h_output', f'decoder_LSTM{idx}_state_c_output']
      tf.identity(self.decoder_model.outputs[idx*2+1], decoder_output_names[-2])
      tf.identity(self.decoder_model.outputs[idx*2+2], decoder_output_names[-1])
      tf.identity(self.decoder_model.inputs[idx*2+1], f'decoder_LSTM{idx}_state_h_input')
      tf.identity(self.decoder_model.inputs[idx*2+2], f'decoder_LSTM{idx}_state_c_input')
    print(f"\t{decoder_output_names}")

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
    plot_model(self.training_model, to_file=f"{self.working_dir}/model.png", show_layer_names=True, show_shapes=True)
    plot_model(self.encoder_model, to_file=f"{self.working_dir}/encoder_model.png", show_layer_names=True, show_shapes=True)
    plot_model(self.decoder_model, to_file=f"{self.working_dir}/decoder_model.png", show_layer_names=True, show_shapes=True)

  def save_no_train(self, width=256):
    self.save_model_plots()
    self.save_summary(width=width)

  def save_model_def(self):
    with open(f"{self.working_dir}/model.json", "w") as file:
      self.model_def["out_tokens"].remove(self.start_token)
      self.model_def["out_tokens"].remove(self.end_token)

      json.dump(self.model_def, file, indent=2)

      self.model_def["out_tokens"].append(self.start_token)
      self.model_def["out_tokens"].append(self.end_token)

  def save(self, width=256):
    self.save_model()
    self.save_model_def()
    self.save_no_train(width)
    self.save_history()
    self.save_report()
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

