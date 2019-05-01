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

  function setSocketStatus(newStatus, color='#000') {
    statusEl.innerText = `Socket status: ${newStatus}`;
    statusEl.style.color = color;
  }

  function notNullOrUndefined(val) {
    return typeof(val) !== 'undefiend' && val !== null;
  }

  let searchParams = (new URL(document.location)).searchParams;
  let ws = new WebSocket(`ws://${searchParams['address'] || 'localhost'}:${searchParams['port'] || 8888}`);
  ws.onopen = (event) => setSocketStatus('connected', '#080');
  ws.onerror = (event) => setSocketStatus(`error - ${event.message}`, '#800');
  ws.close = (event) => setSocketStatus('closed', '#400');

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
        cells[2].innerText = '---';
        cells[3].innerText = '---';
        cells[4].querySelector('input').onclick = (evt) => console.log(task.model_def);
        taskTableBodyEl.appendChild(row);
        if(notNullOrUndefined(task.trained_by)) {
          object.clients[task.trained_by].trained = (object.clients[task.trained_by].trained || 0) + 1;
          cells[2].innerText = task.trained_by;
        }
        if(notNullOrUndefined(task.evaluated_by)) {
          object.clients[task.evaluated_by].evaluted = (object.clients[task.evaluated_by].evaluted || 0) + 1;
          cells[3].innerText = task.evaluated_by;
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
        cells[1].innerText = client.status;
        cells[2].innerText = client.type;
        cells[3].innerText = client.trained || 0;
        cells[4].innerText = client.evaluated || 0;
        cells[5].innerText = client.task ? client.task.name : '---';
        cells[6].innerText = client.task ? client.task.status : '---';
        cells[7].innerText = client.progress !== undefined ? client.progress : '---';
        clientTableBodyEl.appendChild(row);
      }
    }
  };
};

