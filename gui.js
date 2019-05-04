"use strict";

window.onload = function() {
  let statusEl = document.querySelector('#socket-status');
  let updateDateEl = document.querySelector('#last-update');

  let logDateEl = document.querySelector('#last-log-message');
  let logTextareaEl = document.querySelector('#log-textarea');

  let clientTableEl = document.querySelector('#client-table');
  let clientTableBodyEl = clientTableEl.querySelector('tbody');
  let clientTableRowTemplateEl = clientTableBodyEl.querySelector('tr');

  let taskTableEl = document.querySelector('#task-table');
  let taskTableBodyEl = taskTableEl.querySelector('tbody');
  let taskTableRowTemplateEl = taskTableBodyEl.querySelector('tr');

  let addModelsBtn = document.querySelector("#add-models");
  let modelPathInput = document.querySelector("#model-path");
  let addModelDirsBtn = document.querySelector("#add-model-dirs");
  let modelDirPathInput = document.querySelector("#model-dir-path");

  function setSocketStatus(newStatus, color='#000') {
    statusEl.innerText = `Socket status: ${newStatus}`;
    statusEl.style.color = color;
  }

  function notNullOrUndefined(val) {
    return typeof(val) !== 'undefined' && val !== null;
  }

  let searchParams = (new URL(document.location)).searchParams;
  let ws = new WebSocket(`ws://${searchParams['address'] || 'localhost'}:${searchParams['port'] || 8888}`);
  ws.onopen = (event) => setSocketStatus('connected', '#080');
  ws.onerror = (event) => setSocketStatus(`error - ${event.message}`, '#800');
  ws.close = (event) => setSocketStatus('closed', '#400');

  addModelsBtn.onclick = (evt) => ws.send(JSON.stringify({add: {models: [modelPathInput.value]}}));

  addModelDirsBtn.onclick = (evt) => ws.send(JSON.stringify({add: {model_dirs: [modelDirPathInput.value]}}));

  ws.onmessage = (event) => {
    let object = JSON.parse(event.data);
    console.log(new Date(), object);
    if(object.message) {
      logDateEl.innerText = `Last log update ${new Date()}`;
      logTextareaEl.value += `\n[${new Date()}]\n${object.message}\n`;
      logTextareaEl.scrollTop = logTextareaEl.scrollHeight;
    } else {
      updateDateEl.innerText = `Last data update ${new Date()}`;
      
      let taskRow = taskTableRowTemplateEl.cloneNode(true);
      while (taskTableBodyEl.firstChild) {
        taskTableBodyEl.firstChild.remove();
      }
      taskRow.style.display = null;
      for(let taskId in object.tasks) {
        let task = object.tasks[taskId];
        let row = taskRow.cloneNode(true);
        let cells = row.querySelectorAll('td');
        cells[0].innerText = task.name;
        cells[1].innerText = task.status;
        cells[2].innerText = task.status === 'EVALUATING' || task.status === 'TRAINING' ?
            Object.values(object.clients).find(c => c.task && c.task.name == task.name).progress : '---';
        cells[3].innerText = '---';
        cells[4].innerText = '---';
        cells[5].querySelector('input').onclick = (evt) => console.log(task.model_def);
        taskTableBodyEl.appendChild(row);
        if(notNullOrUndefined(task.trained_by)) {
          object.clients[task.trained_by].trained = (object.clients[task.trained_by].trained || 0) + 1;
          cells[3].innerText = task.trained_by;
        }
        if(notNullOrUndefined(task.evaluated_by)) {
          object.clients[task.evaluated_by].evaluated = (object.clients[task.evaluated_by].evaluated || 0) + 1;
          cells[4].innerText = task.evaluated_by;
        }
      }
      
      let clientRow = clientTableRowTemplateEl.cloneNode(true);
      while (clientTableBodyEl.firstChild) {
        clientTableBodyEl.firstChild.remove();
      }
      clientRow.style.display = null;
      for(let clientId in object.clients) {
        let client = object.clients[clientId];
        let row = clientRow.cloneNode(true);
        let cells = row.querySelectorAll('td');
        cells[0].innerText = client.client;
        cells[1].innerText = client.name;
        cells[2].innerText = client.status;
        cells[3].innerText = client.type;
        cells[4].innerText = client.trained || 0;
        cells[5].innerText = client.evaluated || 0;
        cells[6].innerText = client.task ? client.task.name : '---';
        cells[7].innerText = client.task ? client.task.status : '---';
        cells[8].innerText = client.progress !== undefined ? client.progress : '---';
        cells[9].innerText = client.paused !== undefined ? client.paused : '---';
        cells[10].innerText = client.shutdown_scheduled !== undefined ? client.shutdown_scheduled : '---';

        if(notNullOrUndefined(client.paused)) {
          let pause_btn = document.createElement("input");
          pause_btn.type = "button";
          pause_btn.value = "Pause";
          pause_btn.onclick = (evt) => ws.send(JSON.stringify({pause: client.client}));
          if(client.paused) {
            pause_btn.value = "Unpause";
            pause_btn.onclick = (evt) => ws.send(JSON.stringify({unpause: client.client}));
          }
          cells[11].appendChild(pause_btn);
        }

        if(notNullOrUndefined(client.shutdown_scheduled)) {
          let shutdown_btn = document.createElement("input");
          shutdown_btn.type = "button";
          shutdown_btn.value = "Shutdown";
          shutdown_btn.onclick = (evt) => ws.send(JSON.stringify({shutdown: client.client}));
          if(client.shutdown_scheduled) {
            shutdown_btn.value = "Cancel shutdown";
            shutdown_btn.onclick = (evt) => ws.send(JSON.stringify({cancel_shutdown: client.client}));
          }
          cells[11].appendChild(shutdown_btn);
        }

        clientTableBodyEl.appendChild(row);
      }
    }
  };
};

