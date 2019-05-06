import argparse
import copy
import os
import re
import sys
import traceback
from json import JSONEncoder

import tornado
from tornado.httpserver import HTTPServer

from utils import formatter
import pathlib

from tornado import websocket, ioloop
import proto.worker_status_pb2 as WorkerStatus
from proto.worker_message_pb2 import WorkerMessage
import proto.task_status_pb2 as TaskStatus
import proto.task_type_pb2 as TaskType
import proto.worker_type_pb2 as WorkerType
from proto.master_message_pb2 import MasterMessage
from google.protobuf import text_format
import json


class Task(object):
  def __init__(self):
    self.status = TaskStatus.WAITING_TRAIN
    self.name = ""
    self.model_def = {}
    self.trained_by = None
    self.evaluated_by = None


# noinspection PyShadowingBuiltins
class Client(object):
  def __init__(self, client, status, task=None, type=None, progress=None, shutdown_scheduled=False, name=None,
               paused=False, limit=None):
    self.client = client
    self.status = status
    self.task = task
    self.type = type
    self.progress = progress
    self.shutdown_scheduled = shutdown_scheduled
    self.name = name
    self.paused = paused
    self.limit = limit


class SimpleEncoder(JSONEncoder):
  def default(self, o):
    if o.__class__ in [WorkerClient, GuiClient]:
      return o.id
    elif o.__class__ == Task:
      result = copy.deepcopy(o.__dict__)
      result["status"] = TaskStatus.TaskStatus.Name(result["status"])
      return result
    elif o.__class__ == Client:
      result = copy.deepcopy(o).__dict__
      result["status"] = WorkerStatus.WorkerStatus.Name(result["status"])
      result["type"] = None if result["type"] is None else WorkerType.WorkerType.Name(result["type"])
      result["limit"] = None if result["limit"] is None else TaskType.TaskType.Name(result["limit"])
      return result
    else:
      return o.__dict__


# noinspection PyShadowingNames,PyBroadException,PyShadowingBuiltins
class Master(object):
  def __init__(self):
    Master.instance = self
    self.clients = {}
    self.tasks = {}
    self.gui = {}
    self.counter = 0

  def on_new_client(self, client):
    client.id = self.counter
    self.counter += 1
    self.clients[client.id] = Client(client.id, WorkerStatus.PENDING, paused=True)
    self.print("New client(id={}) connected.".format(client.id))

  def print(self, string):
    print(string)
    msg = json.dumps({
      "message": string
    })
    for id, gui in self.gui.items():
      gui.write_message(msg)

  def get_task(self, type, limit):
    first_unknown_train = None
    first_unknown_eval = None
    first_waiting_train = None
    first_waiting_eval = None
    for id, task in self.tasks.items():
      if task.status == TaskStatus.UNKNOWN_WAS_TRAINING and first_unknown_train is None:
        first_unknown_train = task
      elif task.status == TaskStatus.UNKNOWN_WAS_EVALUATING and first_unknown_eval is None:
        first_unknown_eval = task
      elif task.status == TaskStatus.WAITING_TRAIN and first_waiting_train is None:
        first_waiting_train = task
      elif task.status == TaskStatus.WAITING_EVAL and first_waiting_eval is None:
        first_waiting_eval = task
      if None not in [first_waiting_train, first_waiting_eval, first_unknown_eval, first_unknown_train]:
        break

    if all(i is None for i in [first_waiting_eval, first_waiting_train, first_unknown_eval, first_unknown_train]):
      return None

    if limit is not None and limit != TaskType.NONE:
      if limit == TaskType.TRAIN:
        return first_waiting_train if first_waiting_train is not None else first_unknown_train
      else:
        return first_waiting_eval if first_waiting_eval is not None else first_unknown_eval

    if type == WorkerType.CPU:
      return next(
        task for task in [first_waiting_eval, first_unknown_eval, first_waiting_train, first_unknown_train]
        if task is not None
      )
    else:
      return next(
        task for task in [first_waiting_train, first_unknown_train, first_waiting_eval, first_unknown_eval]
        if task is not None
      )

  def update_gui(self):
    for id, gui in self.gui.items():
      data = {
        "clients": self.clients,
        "tasks": self.tasks,
        "gui": self.gui
      }

      gui.write_message(json.dumps(data, cls=SimpleEncoder))

  def send_message(self, client, response):
    self.print(">>>Sending message to client(id={}):\n{}"
               .format(client.id, text_format.MessageToString(response, message_formatter=formatter)))
    client.write_message(response.SerializeToString(), binary=True)

  def handle_idle(self, client, message):
    response = MasterMessage()

    task = self.clients[client.id].task
    if task is not None:
      self.print("Client(id={}) sent idle status, when it still had an assigned job!".format(client.id))
      if task.status == TaskStatus.TRAINING:
        response.task = TaskType.TRAIN
        response.modelDefinition = json.dumps(task.model_def)
      elif task.status == TaskStatus.EVALUATING:
        response.task = TaskType.EVALUATE
      else:
        self.print("Oooops")
        return
      response.taskName = task.name

      self.send_message(client, response)
      return

    # send shutdown task if such is scheduled
    if self.clients[client.id].shutdown_scheduled:
      response.task = TaskType.SHUTDOWN
      response.taskName = "SHUTDOWN"
      self.send_message(client, response)
      return

    if self.clients[client.id].paused:
      response.task = TaskType.NONE
      response.taskName = "NOP"
      self.send_message(client, response)
      return

    task = self.get_task(message.type, self.clients[client.id].limit)
    if task is None:
      self.print("No available tasks")
      response.task = TaskType.NONE
      response.taskName = "NOP"
      self.send_message(client, response)
      return

    if task.status in [TaskStatus.WAITING_TRAIN, TaskStatus.UNKNOWN_WAS_TRAINING]:
      task.status = TaskStatus.TRAINING
      response.task = TaskType.TRAIN
      response.modelDefinition = json.dumps(task.model_def)
    elif task.status in [TaskStatus.WAITING_EVAL, TaskStatus.UNKNOWN_WAS_EVALUATING]:
      file_name = "{}-{}.tar.gz".format(task.name, TaskStatus.TaskStatus.Name(TaskStatus.WAITING_EVAL))
      with open("./work/{}".format(file_name), 'rb') as file:
        response.data = file.read()
      task.status = TaskStatus.EVALUATING
      response.task = TaskType.EVALUATE

    response.taskName = task.name

    self.send_message(client, response)
    self.clients[client.id].task = task

  def handle_done(self, client, message):
    task = self.clients[client.id].task

    if task.status == TaskStatus.TRAINING and message.task == TaskType.TRAIN:
      next_status = TaskStatus.WAITING_EVAL
      task.trained_by = client.id
    elif task.status == TaskStatus.EVALUATING and message.task == TaskType.EVALUATE:
      next_status = TaskStatus.FINISHED
      task.evaluated_by = client.id
    else:
      self.print("Invalid task {} status {} on client(id={}) with task {} when DONE"
                 .format(task.name, TaskStatus.TaskStatus.Name(task.status), client.id,
                         TaskType.TaskType.Name(message.task)))
      self.handle_idle(client, message)
      return

    file_name = "{}-{}.tar.gz".format(task.name, TaskStatus.TaskStatus.Name(next_status))
    if next_status == TaskStatus.FINISHED:
      try:
        pathlib.Path("./work/{}-{}.tar.gz".format(task.name, TaskStatus.TaskStatus.Name(TaskStatus.WAITING_EVAL)))\
          .unlink()
      except Exception:
        self.print("Cannot remove previous archive of {}!".format(file_name))
        traceback.print_exc()
    pathlib.Path("./work/").mkdir(parents=True, exist_ok=True)
    with open("./work/{}".format(file_name), 'wb') as file:
      file.write(message.data)

    task.status = next_status
    self.clients[client.id].task = None
    self.handle_idle(client, message)

  def handle_error(self, client, message):
    self.print("Client(id={}) raised an error!".format(client.id))
    if self.clients[client.id].task is not None:
      task = self.clients[client.id].task
      if task.status == TaskStatus.TRAINING:
        task.status = TaskStatus.UNKNOWN_WAS_TRAINING
        self.print("Client(id={}) had TRAINING task {}, reverting to WAITING_TRAIN".format(client.id, task.name))
      elif task.status == TaskStatus.EVALUATING:
        task.status = TaskStatus.UNKNOWN_WAS_EVALUATING
        self.print("Client(id={}) had EVALUATING task {}, reverting to WAITING_EVAL".format(client.id, task.name))
      else:
        self.print("Task {} has invalid status {} on disconnect, reverting to WAITING_TRAIN"
                   .format(task.name, TaskStatus.TaskStatus.Name(task.status)))
        task.status = TaskStatus.UNKNOWN_WAS_TRAINING
    self.clients[client.id].task = None
    self.handle_idle(client, message)

  def handle_reconnect(self, client, message):
    old_client = None
    old_id = None
    for id, c in self.clients.items():
      if c.name == message.name:
        old_client = c
        old_id = id
        break

    if old_client is None:
      self.print("There was no client with name {}".format(message.name))
      return

    if old_id == client.id:
      self.print("Client(name={}, id={}) was already reassigned.".format(message.name, client.id))
      return

    self.clients[client.id] = old_client
    self.clients[client.id].client = client.id
    self.clients.pop(old_id, None)
    self.print("Client(name={}) reassigned from id={} to id={}.".format(message.name, old_id, client.id))

  def on_client_message(self, client, message):
    self.print("<<<Message from client(id={}):\n{}"
               .format(client.id, text_format.MessageToString(message, message_formatter=formatter)))
    self.clients[client.id].status = message.status
    self.clients[client.id].type = message.type
    self.clients[client.id].progress = message.progress
    self.clients[client.id].name = message.name
    if message.status == WorkerStatus.IDLE:
      self.handle_idle(client, message)
    elif message.status == WorkerStatus.DONE:
      self.handle_done(client, message)
    elif message.status == WorkerStatus.ERROR:
      self.handle_error(client, message)
    elif message.status == WorkerStatus.RECONNECTING:
      self.handle_reconnect(client, message)
    elif message.status in [WorkerStatus.WORKING, WorkerStatus.SHUTTING_DOWN]:
      pass
    else:
      self.print("Got status {} from client(id={}), status handling not implemented!"
                 .format(WorkerStatus.WorkerStatus.Name(message.status), client.id))

    self.update_gui()

  def on_client_disconnect(self, client):
    self.print("Client(id={}) disconnected.".format(client.id))
    if client.id not in self.clients:
      self.print("Client(id={}) not in clients list, was probably reassigned at some point to another id."
                 .format(client.id))
      return
    self.clients[client.id].client = client.id
    self.clients[client.id].status = WorkerStatus.DISCONNECTED
    self.clients[client.id].progress = None
    self.clients[client.id].shutdown_scheduled = None
    self.clients[client.id].paused = None
    self.clients[client.id].progress = None
    if self.clients[client.id].task is not None:
      task = self.clients[client.id].task
      if task.status == TaskStatus.TRAINING:
        task.status = TaskStatus.UNKNOWN_WAS_TRAINING
        self.print("Client(id={}) had TRAINING task {}, reverting to WAITING_TRAIN".format(client.id, task.name))
      elif task.status == TaskStatus.EVALUATING:
        task.status = TaskStatus.UNKNOWN_WAS_EVALUATING
        self.print("Client(id={}) had EVALUATING task {}, reverting to WAITING_EVAL".format(client.id, task.name))
      else:
        self.print(
          "Task {} has invalid status {} after worker disconnect"
            .format(task.name, TaskStatus.TaskStatus.Name(task.status))
        )
    else:
      self.print("Client(id={}) had no task.".format(client.id))
    self.clients[client.id].task = None
    self.update_gui()

  def on_new_gui_client(self, gui):
    gui.id = self.counter
    self.counter += 1
    self.gui[gui.id] = gui
    self.update_gui()

  def on_gui_client_disconnect(self, gui):
    self.gui.pop(gui.id, None)

  def add_models(self, models, repeats, epochs, batch_size, present_models={}):
    for model in models:
      for i in range(0, repeats):
        task = Task()
        task.status = TaskStatus.WAITING_TRAIN
        with open(model) as file:
          task.model_def = json.load(file)
        task.model_def["epochs"] = epochs
        task.model_def["batch_size"] = batch_size
        task.name = "{}-{}".format(task.model_def["name"], i)
        if task.model_def["name"] in present_models and str(i) in present_models[task.model_def["name"]]:
          task.status = present_models[task.model_def["name"]][str(i)]
          self.print("Task {} already exists with status {}".format(task.name, TaskStatus.TaskStatus.Name(task.status)))
        elif task.name in self.tasks:
          self.print("Task {} already exists with status {}".format(
            task.name, TaskStatus.TaskStatus.Name(self.tasks[task.name].status)))
          continue
        else:
          self.print("Added task {}".format(task.name))
        self.tasks[task.name] = task


master_instance = Master()


# noinspection PyBroadException,PyShadowingNames
class WorkerClient(websocket.WebSocketHandler):
  def data_received(self, chunk):
    pass

  def on_message(self, msg):
    try:
      message = WorkerMessage()
      message.ParseFromString(msg)
      master_instance.on_client_message(self, message)
    except Exception:
      traceback.print_exc()

  def open(self):
    try:
      master_instance.on_new_client(self)
    except Exception:
      traceback.print_exc()

  def on_close(self):
    try:
      master_instance.on_client_disconnect(self)
    except Exception:
      traceback.print_exc()

  def check_origin(self, origin):
    return True


parser = argparse.ArgumentParser()

parser.add_argument("--gui_address", default="", type=str)
parser.add_argument("--gui_port", default=8888, type=int)
parser.add_argument("--worker_address", default="", type=str)
parser.add_argument("--worker_port", default=8000, type=int)
parser.add_argument("--models", nargs='+', type=str, default=[])
parser.add_argument("--model_dirs", nargs='+', type=str, default=[])
parser.add_argument("--repeat", default=5, type=int)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--batch_size", default=64, type=int)


args = parser.parse_args()


def collect_models(models=[], model_dirs=[]):
  mdls = [] + models
  for directory in model_dirs:
    files = os.listdir(directory)
    for file in files:
      if file.endswith(".json"):
        mdls.append("{}/{}".format(directory, file))
  return mdls


# noinspection PyBroadException,PyShadowingNames
class GuiClient(websocket.WebSocketHandler):
  def data_received(self, chunk):
    pass

  def on_message(self, message):
    try:
      msg = json.loads(message)
      if "shutdown" in msg:
        master_instance.clients[msg["shutdown"]].shutdown_scheduled = True
      if "cancel_shutdown" in msg:
        master_instance.clients[msg["cancel_shutdown"]].shutdown_scheduled = False
      if "pause" in msg:
        master_instance.clients[msg["pause"]].paused = True
      if "unpause" in msg:
        master_instance.clients[msg["unpause"]].paused = False
      if "add" in msg:
        models = [] if "models" not in msg["add"] else msg["add"]["models"]
        model_dirs = [] if "model_dirs" not in msg["add"] else msg["add"]["model_dirs"]
        master_instance.add_models(collect_models(models, model_dirs), args.repeat, args.epochs, args.batch_size)
      if "limit" in msg:
        master_instance.clients[msg["limit"]["client"]].limit = TaskType.TaskType.Value(msg["limit"]["value"])
      master_instance.update_gui()
    except Exception:
      err = {"error": traceback.format_exc()}
      self.write_message(json.dumps(err))

  def open(self):
    try:
      master_instance.on_new_gui_client(self)
    except Exception:
      traceback.print_exc()

  def on_close(self):
    try:
      master_instance.on_gui_client_disconnect(self)
    except Exception:
      traceback.print_exc()

  def check_origin(self, origin):
    return True


worked_models = os.listdir("./work") if os.path.isdir("./work") else []
present_models = {}
for idx, value in enumerate(worked_models):
  match = re.match(r'(.*)-([0-9]*)-([A-Z_]*)\.tar\.gz', value)
  if match is not None and TaskStatus.TaskStatus.Value(match.groups()[2]) is not None:
    groups = match.groups()
    name = groups[0]
    repeat = groups[1]
    status = groups[2]
    if name not in present_models:
      present_models[name] = {}
    present_models[name][repeat] = TaskStatus.TaskStatus.Value(status)


models = collect_models(args.models, args.model_dirs)

if len(models) == 0:
  print("No tasks!")
  sys.exit(-1)

master_instance.add_models(models, args.repeat, args.epochs, args.batch_size, present_models)

gui_app = tornado.web.Application([(r"/", GuiClient)], websocket_max_message_size=1024*1024*1024)
worker_app = tornado.web.Application([(r"/", WorkerClient)], websocket_max_message_size=1024*1024*1024)

worker_server = HTTPServer(worker_app, max_buffer_size=1024*1024*1024)

gui_app.listen(args.gui_port, address=args.gui_address)
worker_server.listen(args.worker_port, address=args.worker_address)

tornado.ioloop.IOLoop.instance().start()
