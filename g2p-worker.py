import json
import os
import pathlib
import shutil
import threading
import traceback
from io import BytesIO
from keras import backend as K

import websocket
from keras.callbacks import LambdaCallback
from websocket import ABNF

from Seq2Seq import Seq2Seq
from proto.master_message_pb2 import MasterMessage
from google.protobuf import text_format

from utils import formatter, load_records

import time
from proto.worker_message_pb2 import WorkerMessage
import proto.worker_status_pb2 as WorkerStatus
import proto.worker_type_pb2 as WorkerType
import proto.task_type_pb2 as TaskType
import tensorflow as tf
import tarfile
import argparse
import random
import socket


parser = argparse.ArgumentParser()
parser.add_argument("--address", default="localhost", type=str)
parser.add_argument("--port", default=8000, type=int)
parser.add_argument("--trace", default=False, type=bool)
parser.add_argument("--name", default=socket.gethostname(), type=str)
args = parser.parse_args()

status = WorkerStatus.IDLE
task = TaskType.NONE
if tf.test.gpu_device_name():
  device = WorkerType.GPU
else:
  device = WorkerType.CPU
data = None
name = "{}-{}".format(args.name, random.randint(0, 1000000))
work_dir = "g2p-worker-{}".format(name)
progress = 0
work = True
reconnected = False


def compress_and_set_result(model_path):
  global status
  global data
  io_output = BytesIO()
  with tarfile.open(fileobj=io_output, mode="w:gz") as tar:
    tar.add(model_path, arcname=os.path.basename(model_path))

  data = io_output.getvalue()
  status = WorkerStatus.DONE

  try:
    shutil.rmtree(work_dir)
  except FileNotFoundError:
    pass


def set_progress(value):
  global progress
  progress = value


# noinspection PyBroadException
def train(message):
  K.clear_session()
  global status
  global data
  global task
  global progress
  data = None
  status = WorkerStatus.WORKING
  task = TaskType.TRAIN
  progress = 0
  try:
    model_def = json.loads(message.modelDefinition)
    records = load_records()
    model_path = "{}/{}".format(work_dir, message.taskName)
    try:
      shutil.rmtree(model_path)
    except FileNotFoundError:
      pass
    model = Seq2Seq(model_def, working_dir=model_path)
    model.train(records, epochs=model_def["epochs"], batch_size=model_def["batch_size"],
                callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: set_progress(epoch+1))])
    model.save_model()
    model.save_model_def()
    model.save_no_train()
    model.save_history()
    model.save_for_inference_tf()

    compress_and_set_result(model_path)
  except Exception:
    status = WorkerStatus.ERROR
    task = TaskType.NONE
    progress = 0
    traceback.print_exc()


# noinspection PyBroadException
def validate(message):
  K.clear_session()
  global status
  global data
  global task
  global progress
  data = None
  status = WorkerStatus.WORKING
  task = TaskType.EVALUATE
  progress = 0
  try:
    model_path = "{}/{}".format(work_dir, message.taskName)
    try:
      shutil.rmtree(model_path)
    except FileNotFoundError:
      pass
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    io_input = BytesIO(message.data)
    with tarfile.open(fileobj=io_input, mode="r:gz") as tar:
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(tar, work_dir)
    model = Seq2Seq(load=True, working_dir=model_path)
    model.save_full_report()
    progress = 1
    model.evaluate_checkpoints(progress=lambda p: set_progress(p+1), data_type=['validate'])

    compress_and_set_result(model_path)
  except Exception:
    status = WorkerStatus.ERROR
    task = TaskType.NONE
    progress = 0
    traceback.print_exc()


def on_message(ws, data):
  global status
  global task
  global work
  global progress
  message = MasterMessage()
  message.ParseFromString(data)
  print("<<<Got message from master:\n{}".format(text_format.MessageToString(message, message_formatter=formatter)))
  if message.task in [TaskType.TRAIN, TaskType.EVALUATE] \
     and status not in [WorkerStatus.DONE, WorkerStatus.IDLE, WorkerStatus.ERROR]:
    print("Current task still not finished, ignoring!")
    return
  if message.task == TaskType.TRAIN:
    threading.Thread(target=train, args=[message]).start()
  elif message.task == TaskType.EVALUATE:
    threading.Thread(target=validate, args=[message]).start()
  elif message.task == TaskType.NONE:
    status = WorkerStatus.IDLE
    task = TaskType.NONE
    progress = 0
  elif message.task == TaskType.SHUTDOWN:
    status = WorkerStatus.SHUTTING_DOWN
    task = TaskType.NONE
    progress = 0
    work = False
  else:
    print("Got unknown task form master: {}".format(text_format.MessageToString(message.task)))


def on_error(ws, error):
  print(error)


def on_close(ws):
  print("### closed ###")


def on_open(ws):
  message = WorkerMessage()
  message.status = status
  message.type = device
  print("\n>>>Sending message to master:\n{}"
        .format(text_format.MessageToString(message, message_formatter=formatter)))
  ws.send(message.SerializeToString(), ABNF.OPCODE_BINARY)


websocket.enableTrace(args.trace)
ws = websocket.WebSocketApp("ws://{}:{}/".format(args.address, args.port),
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)
ws.on_open = on_open


def heartbeat(ws):
  global reconnected
  sleep = 5
  while work:
    time.sleep(sleep)
    if ws is not None:
      if sleep == 5:
        sleep = 60
      message = WorkerMessage()
      if reconnected:
        message.status = WorkerStatus.RECONNECTING
        message.task = task
        message.type = device
        message.name = name
        print("\n>>>Sending message to master:\n{}"
              .format(text_format.MessageToString(message, message_formatter=formatter)))
        ws.send(message.SerializeToString(), ABNF.OPCODE_BINARY)
        reconnected = False
        continue

      message = WorkerMessage()
      message.status = status
      message.type = device
      message.task = task
      message.progress = progress
      message.name = name
      if status == WorkerStatus.DONE:
        message.data = data
      print("\n>>>Sending message to master:\n{}"
            .format(text_format.MessageToString(message, message_formatter=formatter)))
      ws.send(message.SerializeToString(), ABNF.OPCODE_BINARY)
  ws.close()


thrd = threading.Thread(target=heartbeat, args=[ws], daemon=True)
thrd.start()
ws.run_forever()
while work:
  print("Connection dropped, reconnecting...")
  reconnected = True
  ws.run_forever()
thrd.join()
