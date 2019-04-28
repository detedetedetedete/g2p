import argparse
import copy
import os
import sys
import threading
import traceback
from json import JSONEncoder

from utils import formatter
import pathlib

from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
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


class SimpleEncoder(JSONEncoder):
  def default(self, o):
    if o.__class__ in [WorkerClient, GuiClient]:
      return o.id
    elif o.__class__ == Task:
      result = copy.deepcopy(o.__dict__)
      result["status"] = TaskStatus.TaskStatus.Name(result["status"])
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
    self.clients[client.id] = {
      "client": client,
      "status": WorkerStatus.PENDING,
      "task": None
    }
    self.print("New client(id={}) connected.".format(client.id))

  def print(self, string):
    print(string)
    msg = json.dumps({
      "message": string
    })
    for gui in self.gui:
      gui.sendMessage(msg)

  def get_task(self, type):
    first_unknown_train = None
    first_unknown_eval = None
    first_waiting_train = None
    first_waiting_eval = None
    for id, task in self.tasks.items():
      if task.status == TaskStatus.UNKNOWN_WAS_TRAINING:
        first_unknown_train = task
      elif task.status == TaskStatus.UNKNOWN_WAS_EVALUATING:
        first_unknown_eval = task
      elif task.status == TaskStatus.WAITING_TRAIN:
        first_waiting_train = task
      elif task.status == TaskStatus.WAITING_EVAL:
        first_waiting_eval = task
      if None not in [first_waiting_eval, first_waiting_train, first_unknown_eval, first_unknown_train]:
        break

    if all(i is None for i in [first_waiting_eval, first_waiting_train, first_unknown_eval, first_unknown_train]):
      return None

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

      gui.sendMessage(json.dumps(data, cls=SimpleEncoder))

  def send_message(self, client, response):
    self.print(">>>Sending message to client(id={}):\n{}"
               .format(client.id, text_format.MessageToString(response, message_formatter=formatter)))
    client.sendMessage(response.SerializeToString())

  def handle_idle(self, client, message):
    response = MasterMessage()

    task = self.clients[client.id]["task"]
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
    task = self.get_task(message.type)
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
    self.clients[client.id]["task"] = task

  def handle_done(self, client, message):
    task = self.clients[client.id]["task"]

    if task.status == TaskStatus.TRAINING and message.task == TaskType.TRAIN:
      next_status = TaskStatus.WAITING_EVAL
    elif task.status == TaskStatus.EVALUATING and message.task == TaskType.EVALUATE:
      next_status = TaskStatus.FINISHED
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
    self.clients[client.id]["task"] = None
    self.handle_idle(client, message)

  def handle_error(self, client, message):
    self.print("Client(id={}) raised an error!".format(client.id))
    if self.clients[client.id]["task"] is not None:
      task = self.clients[client.id]["task"]
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
    self.clients[client.id]["task"] = None
    self.handle_idle(client, message)

  def on_client_message(self, client, message):
    self.print("<<<Message from client(id={}):\n{}"
               .format(client.id, text_format.MessageToString(message, message_formatter=formatter)))
    self.clients[client.id]["status"] = message.status
    if message.status == WorkerStatus.IDLE:
      self.handle_idle(client, message)
    elif message.status == WorkerStatus.DONE:
      self.handle_done(client, message)
    elif message.status == WorkerStatus.ERROR:
      self.handle_error(client, message)
    elif message.status == WorkerStatus.WORKING:
      pass
    else:
      self.print("Got status {} from client(id={}), status handling not implemented!"
                 .format(WorkerStatus.WorkerStatus.Name(message.status), client.id))

    self.update_gui()

  def on_client_disconnect(self, client):
    self.print("Client(id={}) disconnected.".format(client.id))
    self.clients[client.id]["status"] = WorkerStatus.DISCONNECTED
    if self.clients[client.id]["task"] is not None:
      task = self.clients[client.id]["task"]
      if task.status == TaskStatus.TRAINING:
        task.status = TaskStatus.UNKNOWN_WAS_TRAINING
        self.print("Client(id={}) had TRAINING task {}, reverting to WAITING_TRAIN".format(client.id, task.name))
      elif task.status == TaskStatus.EVALUATING:
        task.status = TaskStatus.UNKNOWN_WAS_TRAINING
        self.print("Client(id={}) had EVALUATING task {}, reverting to WAITING_EVAL".format(client.id, task.name))
      else:
        self.print(
          "Task {} has invalid status {} after worker disconnect"
            .format(task.name, TaskStatus.TaskStatus.Name(task.status))
        )
    else:
      self.print("Client(id={}) had no task.".format(client.id))
    self.clients[client.id]["task"] = None

  def on_new_gui_client(self, gui):
    gui.id = self.counter
    self.counter += 1
    self.gui[gui.id] = gui
    self.update_gui()

  def on_gui_client_disconnect(self, gui):
    self.gui.pop(gui.id, None)


master_instance = Master()


# noinspection PyBroadException,PyShadowingNames
class WorkerClient(WebSocket):
  def __init__(self, server, sock, address):
    super().__init__(server, sock, address)
    self.id = None

  def handleMessage(self):
    try:
      message = WorkerMessage()
      message.ParseFromString(self.data)
      master_instance.on_client_message(self, message)
    except Exception:
      traceback.print_exc()

  def handleConnected(self):
    try:
      master_instance.on_new_client(self)
    except Exception:
      traceback.print_exc()

  def handleClose(self):
    try:
      master_instance.on_client_disconnect(self)
    except Exception:
      traceback.print_exc()


# noinspection PyBroadException,PyShadowingNames
class GuiClient(WebSocket):
  def __init__(self, server, sock, address):
    super().__init__(server, sock, address)
    self.id = None

  def handleConnected(self):
    try:
      master_instance.on_new_gui_client(self)
    except Exception:
      traceback.print_exc()

  def handleClose(self):
    try:
      master_instance.on_gui_client_disconnect(self)
    except Exception:
      traceback.print_exc()


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

models = args.models
for directory in args.model_dirs:
  files = os.listdir(directory)
  for file in files:
    if file.endswith(".json"):
      models.append("{}/{}".format(directory, file))

if len(models) == 0:
  print("No tasks!")
  sys.exit(-1)

for model in models:
  for i in range(0, args.repeat):
    task = Task()
    task.status = TaskStatus.WAITING_TRAIN
    with open(model) as file:
      task.model_def = json.load(file)
    task.model_def["epochs"] = args.epochs
    task.model_def["batch_size"] = args.batch_size
    task.name = "{}-{}".format(task.model_def["name"], i)
    print("Added task {}".format(task.name))
    master_instance.tasks[task.name] = task

gui_server = SimpleWebSocketServer(args.gui_address, args.gui_port, GuiClient)
threading.Thread(target=lambda: gui_server.serveforever(), args=[]).start()


server = SimpleWebSocketServer(args.worker_address, args.worker_port, WorkerClient)
server.serveforever()
