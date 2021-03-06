# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: task_status.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='task_status.proto',
  package='g2p',
  syntax='proto2',
  serialized_pb=_b('\n\x11task_status.proto\x12\x03g2p*\x93\x01\n\nTaskStatus\x12\x11\n\rWAITING_TRAIN\x10\x00\x12\x0c\n\x08TRAINING\x10\x01\x12\x10\n\x0cWAITING_EVAL\x10\x02\x12\x0e\n\nEVALUATING\x10\x03\x12\x18\n\x14UNKNOWN_WAS_TRAINING\x10\x04\x12\x1a\n\x16UNKNOWN_WAS_EVALUATING\x10\x05\x12\x0c\n\x08\x46INISHED\x10\x06')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_TASKSTATUS = _descriptor.EnumDescriptor(
  name='TaskStatus',
  full_name='g2p.TaskStatus',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='WAITING_TRAIN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TRAINING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WAITING_EVAL', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EVALUATING', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_WAS_TRAINING', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_WAS_EVALUATING', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FINISHED', index=6, number=6,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=27,
  serialized_end=174,
)
_sym_db.RegisterEnumDescriptor(_TASKSTATUS)

TaskStatus = enum_type_wrapper.EnumTypeWrapper(_TASKSTATUS)
WAITING_TRAIN = 0
TRAINING = 1
WAITING_EVAL = 2
EVALUATING = 3
UNKNOWN_WAS_TRAINING = 4
UNKNOWN_WAS_EVALUATING = 5
FINISHED = 6


DESCRIPTOR.enum_types_by_name['TaskStatus'] = _TASKSTATUS


# @@protoc_insertion_point(module_scope)
