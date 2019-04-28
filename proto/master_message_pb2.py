# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: master_message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import proto.task_type_pb2 as task__type__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='master_message.proto',
  package='g2p',
  syntax='proto2',
  serialized_pb=_b('\n\x14master_message.proto\x12\x03g2p\x1a\x0ftask_type.proto\"e\n\rMasterMessage\x12\x1b\n\x04task\x18\x01 \x02(\x0e\x32\r.g2p.TaskType\x12\x17\n\x0fmodelDefinition\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\x10\n\x08taskName\x18\x04 \x02(\t')
  ,
  dependencies=[task__type__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_MASTERMESSAGE = _descriptor.Descriptor(
  name='MasterMessage',
  full_name='g2p.MasterMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='task', full_name='g2p.MasterMessage.task', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='modelDefinition', full_name='g2p.MasterMessage.modelDefinition', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='g2p.MasterMessage.data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='taskName', full_name='g2p.MasterMessage.taskName', index=3,
      number=4, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=46,
  serialized_end=147,
)

_MASTERMESSAGE.fields_by_name['task'].enum_type = task__type__pb2._TASKTYPE
DESCRIPTOR.message_types_by_name['MasterMessage'] = _MASTERMESSAGE

MasterMessage = _reflection.GeneratedProtocolMessageType('MasterMessage', (_message.Message,), dict(
  DESCRIPTOR = _MASTERMESSAGE,
  __module__ = 'master_message_pb2'
  # @@protoc_insertion_point(class_scope:g2p.MasterMessage)
  ))
_sym_db.RegisterMessage(MasterMessage)


# @@protoc_insertion_point(module_scope)
