import argparse
import re
import sys

import xlsxwriter
import os


parser = argparse.ArgumentParser()

parser.add_argument("--work_dirs", nargs='+', type=str, default=[])
parser.add_argument("--name", default='report.xslx', type=str)

args = parser.parse_args()

if len(args.work_dirs) == 0:
  print("No work dirs specified!")
  sys.exit(1)

work = []
for wdir in args.work_dirs:
  models = os.listdir(wdir)
  work += ["{}/{}".format(wdir, f) for f in models if os.path.isdir("{}/{}/checkpoints".format(wdir, f))]

# name -> bandymas -> epoch -> accuracy
results = {}
max_errors = 0
for instance in work:
  match = re.match(r'(.*?)-([0-9]+)$', instance)
  if match is None:
    print("{} is not a valid name for a training result!".format(instance))
    continue
  groups = match.groups()
  name = groups[0]
  bd = groups[1]

  for log in os.listdir("{}/checkpoints".format(instance)):
    if not log.endswith(".log"):
      continue
    match = re.match(r'weights\.([0-9]+)-', log)
    if match is None:
      print("Invalid log file name {}".format(log))
      continue
    epoch = match.groups()[0]
    if name not in results:
      results[name] = {}
      results[name]["best"] = {
        "idx": None,
        "value": 0
      }
    if bd not in results[name]:
      results[name][bd] = {}
      results[name][bd]["best"] = {
        "idx": epoch,
        "value": 0
      }
    if epoch not in results[name][bd]:
      results[name][bd][epoch] = {}
    with open("{}/checkpoints/{}".format(instance, log)) as file:
      val = list(file)
      start = False
      for line in val:
        if not start and re.match(r'-+Validation set-+', line):
          start = True
        if not start:
          continue
        if re.match(r'-+Full set-+', line):
          break
        match = re.match(r'\tHad ([0-9]+) mistakes: [0-9]+ \(([0-9.]+)%\)', line)
        if match is None:
          continue
        g = match.groups()
        err_count = g[0]
        errs = g[1]
        results[name][bd][epoch][err_count] = errs
        if int(err_count) == 0 and float(results[name][bd]["best"]["value"]) < float(errs):
          results[name][bd]["best"]["idx"] = epoch
          results[name][bd]["best"]["value"] = errs
        if int(err_count) == 0 and float(results[name]["best"]["value"]) < float(errs):
          results[name]["best"]["idx"] = bd
          results[name]["best"]["value"] = errs
        if int(err_count) > max_errors:
          max_errors = int(err_count)


def get_name(string):
  return re.match(r'.*?([^/]+)$', string).groups()[0]


workbook = xlsxwriter.Workbook(args.name)
worksheet = workbook.add_worksheet('report')
general_chart = workbook.add_chart({'type': 'column'})
bold_format = workbook.add_format({'bold': True})
accent_format = workbook.add_format({'bold': True, 'color': 'red'})
title_format = workbook.add_format({'bold': True, 'align': 'center'})
category_format = workbook.add_format({'bold': True, 'align': 'center'})

offset = 35
for r_idx, (name, bandymai) in enumerate(sorted(results.items(), key=lambda x: x[0])):
  chart = workbook.add_chart({'type': 'column'})
  chart.set_size({'x_scale': 2.2, 'y_scale': 2.2})
  chart.set_title({'name': 'Klaidų pasiskirstymas'})
  chart.set_x_axis({'name': 'Klaidų kiekis'})
  chart.set_y_axis({'name': 'Dalis, %', 'min': 0, 'max': 100})
  chart2 = workbook.add_chart({'type': 'column'})
  chart2.set_size({'x_scale': 1, 'y_scale': 2.2})
  chart2.set_title({'name': 'Klaidų pasiskirstymas'})
  chart2.set_x_axis({'name': 'Klaidų kiekis'})
  chart2.set_y_axis({'name': 'Dalis, %', 'min': 90, 'max': 100})
  worksheet.merge_range(offset * r_idx, 1, offset * r_idx, 5, get_name(name), title_format)
  worksheet.merge_range(offset*len(results) + r_idx, 1, offset*len(results) + r_idx, 5, get_name(name), category_format)
  for i in range(10, 301, 10):
    worksheet.write(int(offset * r_idx + 1 + i / 10), 0, i, bold_format)
  for i in range(0, max_errors+1):
    worksheet.write(offset * len(results) - 1, 6 + i, i, bold_format)
    worksheet.write(offset * r_idx + 2 + i, 7, i, bold_format)

  best_avg = 0
  for b_idx, (b_name, epochs) in enumerate(sorted(bandymai.items(), key=lambda x: x[0])):
    if b_name == 'best':
      continue
    worksheet.write(offset*r_idx + 1, b_idx + 1, b_name, title_format)
    worksheet.write(offset*r_idx + 1, b_idx + 8, b_name, title_format)

    for e_idx, (e_name, accuracies) in enumerate(sorted(epochs.items(), key=lambda x: x[0])):
      if e_name == 'best':
        continue
      for a_idx in range(0, max_errors+1):
        if a_idx == 0:
          if epochs["best"]["idx"] == e_name and bandymai["best"]["idx"] == b_name:
            worksheet.write(offset * r_idx + 2 + e_idx, b_idx + 1,
                            0 if str(a_idx) not in accuracies else float(accuracies[str(a_idx)]), accent_format)
            best_avg += float(accuracies[str(a_idx)])
          elif epochs["best"]["idx"] == e_name:
            worksheet.write(offset * r_idx + 2 + e_idx, b_idx + 1,
                            0 if str(a_idx) not in accuracies else float(accuracies[str(a_idx)]), bold_format)
            best_avg += float(accuracies[str(a_idx)])
          else:
            worksheet.write(offset * r_idx + 2 + e_idx, b_idx + 1,
                            0 if str(a_idx) not in accuracies else float(accuracies[str(a_idx)]))
        if epochs["best"]["idx"] == e_name:
          if a_idx == 0:
            chart.add_series({
              'name': ['report', offset*r_idx + 1, b_idx + 8],
              'categories': ['report', offset*r_idx + 2, 7, offset*r_idx + 2 + max_errors, 7],
              'values': ['report', offset*r_idx + 2, b_idx + 8, offset*r_idx + 2 + max_errors, b_idx + 8]
            })
            chart2.add_series({
              'name': ['report', offset * r_idx + 1, b_idx + 8],
              'categories': ['report', offset * r_idx + 2, 7, offset * r_idx + 2, 7],
              'values': ['report', offset * r_idx + 2, b_idx + 8, offset * r_idx + 2, b_idx + 8]
            })
          worksheet.write(offset * r_idx + 2 + a_idx, b_idx + 8,
                          0 if str(a_idx) not in accuracies else float(accuracies[str(a_idx)]))
          if bandymai["best"]["idx"] == b_name:
            worksheet.write(offset * len(results) + r_idx, 6 + a_idx,
                          0 if str(a_idx) not in accuracies else float(accuracies[str(a_idx)]))
            if a_idx == 0:
              general_chart.add_series({
                'name': ['report', offset * len(results) + r_idx, 1],
                'categories': ['report', offset * len(results) - 1, 6, offset * len(results) - 1, 6 + 5],
                'values': ['report', offset * len(results) + r_idx, 6, offset * len(results) + r_idx, 6 + 5]
              })

  best_avg /= len(bandymai) - 1
  worksheet.write(offset * len(results) + r_idx, 0, round(best_avg, 2), bold_format)

  worksheet.insert_chart(offset*r_idx + 1, 14, chart)
  worksheet.insert_chart(offset*r_idx + 1, 31, chart2)
general_chart.set_size({'x_scale': 5, 'y_scale': 2.2})
general_chart.set_title({'name': 'Klaidų pasiskirstymas'})
general_chart.set_x_axis({'name': 'Klaidų kiekis'})
general_chart.set_y_axis({'name': 'Dalis, %', 'min': 0, 'max': 100})
worksheet.insert_chart((offset+1) * len(results), 1, general_chart)
workbook.close()
