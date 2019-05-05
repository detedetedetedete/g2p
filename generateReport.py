import re
import xlsxwriter
import os


work = os.listdir("work")
work = [f for f in work if os.path.isdir("work/{}/checkpoints".format(f))]

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

  for log in os.listdir("work/{}/checkpoints".format(instance)):
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
    with open("work/{}/checkpoints/{}".format(instance, log)) as file:
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


workbook = xlsxwriter.Workbook('report.xlsx')
worksheet = workbook.add_worksheet('report')
bold_format = workbook.add_format({'bold': True})
accent_format = workbook.add_format({'bold': True, 'color': 'red'})
title_format = workbook.add_format({'bold': True, 'align': 'center'})

offset = 35
for r_idx, (name, bandymai) in enumerate(sorted(results.items(), key=lambda x: x[0])):
  chart = workbook.add_chart({'type': 'column'})
  chart.set_size({'x_scale': 2.2, 'y_scale': 2.2})
  chart.set_title({'name': 'Klaidų pasiskirstymas'})
  chart.set_x_axis({'name': 'Klaidų kiekis'})
  chart.set_y_axis({'name': 'Dalis, %', 'min': 0, 'max': 100})
  worksheet.merge_range(offset * r_idx, 1, offset * r_idx, 5, name, title_format)
  for i in range(10, 301, 10):
    worksheet.write(int(offset * r_idx + 1 + i / 10), 0, i, bold_format)
  for i in range(0, max_errors+1):
    worksheet.write(offset * r_idx + 2 + i, 7, i, bold_format)

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
          elif epochs["best"]["idx"] == e_name:
            worksheet.write(offset * r_idx + 2 + e_idx, b_idx + 1,
                            0 if str(a_idx) not in accuracies else float(accuracies[str(a_idx)]), bold_format)
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
          worksheet.write(offset * r_idx + 2 + a_idx, b_idx + 8,
                          0 if str(a_idx) not in accuracies else float(accuracies[str(a_idx)]))
  worksheet.insert_chart(offset*r_idx + 1, 14, chart)


workbook.close()