# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: worker_message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import proto.worker_status_pb2 as worker__status__pb2
import proto.worker_type_pb2 as worker__type__pb2
import proto.task_type_pb2 as task__type__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='worker_message.proto',
  package='g2p',
  syntax='proto2',
  serialized_pb=_b('\n\x14worker_message.proto\x12\x03g2p\x1a\x13worker_status.proto\x1a\x11worker_type.proto\x1a\x0ftask_type.proto\"\x8e\x01\n\rWorkerMessage\x12!\n\x06status\x18\x01 \x02(\x0e\x32\x11.g2p.WorkerStatus\x12\x1b\n\x04task\x18\x02 \x02(\x0e\x32\r.g2p.TaskType\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\x1d\n\x04type\x18\x04 \x02(\x0e\x32\x0f.g2p.WorkerType\x12\x10\n\x08progress\x18\x05 \x01(\x05')
  ,
  dependencies=[worker__status__pb2.DESCRIPTOR,worker__type__pb2.DESCRIPTOR,task__type__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_WORKERMESSAGE = _descriptor.Descriptor(
  name='WorkerMessage',
  full_name='g2p.WorkerMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='g2p.WorkerMessage.status', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='task', full_name='g2p.WorkerMessage.task', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='g2p.WorkerMessage.data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='g2p.WorkerMessage.type', index=3,
      number=4, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='progress', full_name='g2p.WorkerMessage.progress', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=229,
)

_WORKERMESSAGE.fields_by_name['status'].enum_type = worker__status__pb2._WORKERSTATUS
_WORKERMESSAGE.fields_by_name['task'].enum_type = task__type__pb2._TASKTYPE
_WORKERMESSAGE.fields_by_name['type'].enum_type = worker__type__pb2._WORKERTYPE
DESCRIPTOR.message_types_by_name['WorkerMessage'] = _WORKERMESSAGE

WorkerMessage = _reflection.GeneratedProtocolMessageType('WorkerMessage', (_message.Message,), dict(
  DESCRIPTOR = _WORKERMESSAGE,
  __module__ = 'worker_message_pb2'
  # @@protoc_insertion_point(class_scope:g2p.WorkerMessage)
  ))
_sym_db.RegisterMessage(WorkerMessage)


# @@protoc_insertion_point(module_scope)
