import json
import re


def formatter(message, indent, as_one_line):
  result = ''
  for field, value in message.ListFields():
    if field.enum_type is None:
      if field.type == field.TYPE_BYTES:
        result += '{}: <bytes> {:.2f} MB\n'.format(field.name, len(value)/1000000)
      else:
        result += '{}: {}\n'.format(field.name, value)
    else:
      result += '{}: {}\n'.format(field.name, field.enum_type.values_by_number[value].name)
  return result


def split_record(rec):
  parts = re.split(' {3,}', rec)
  return parts[0], parts[1].split(' ')


def load_model(path):
  with open(path) as model_def_file:
    return json.load(model_def_file)


def load_records():
  return dict(
    [split_record(record) for record in filter(None, open('./g2p.dict', 'r', encoding='utf-8').read().split('\n'))]
  )