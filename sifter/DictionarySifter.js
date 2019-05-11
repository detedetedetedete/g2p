const readline = require('readline');
const fs = require('fs');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});


var dictionary = new Set();
var states = {
    console: 0,
    raw: 1
};

var commands = {
    raw: enableRaw,
    file: handleFile,
    dump: handleDump,
    clear: handleClear,
    count: handleCount,
    exit: handleExit
};

var state = states.console;

rl.on('line', (input) => {
    if(state === states.console) {
        handleConsole(input);
    } else if(state === states.raw) {
        handleRaw(dictionary, input);
    }
});

rl.on('SIGINT', () => {
    if(state === states.raw) {
        console.log(`>>> Exiting raw mode.`);
        state = states.console;
    } else {
        console.log(`>>> Enter exit to quit.`);
    }
});

function handleConsole(input) {
    input = input.toLowerCase();
    if(input.indexOf(' ') === -1) {
        if(commands[input]) {
            commands[input](dictionary);
        } else {
            console.log(`>>> Unknown command.`);
        }
    } else {
        if(commands[input.substr(0,input.indexOf(' '))]) {
            commands[input.substr(0,input.indexOf(' '))](dictionary, input.substr(input.indexOf(' ')+1));
        } else {
            console.log(`>>> Unknown command.`);
        }
    }
}

function handleRaw(dict, input, showCount=true) {
    // first split removes words, that have non-lithuanian letters, other non-alpha symbols.
    var cleanArray = input.split(/\b\w*?[^AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽaąbcčdeęėfghiįyjklmnoprsštuųūvzž \.,].*?\b/).join(' ').split(/[^AĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽaąbcčdeęėfghiįyjklmnoprsštuųūvzž]/);
    cleanArray = cleanArray.filter(element => element.length);
    cleanArray.forEach(element => dict.add(element.toLowerCase()));
    if(showCount) {
        commands.count(dict);
    }
}

function enableRaw() {
    console.log(`>>> Entering raw reading mode. To stop, press CTRL+C`);
    state = states.raw;
}

function handleFile(dict, file) {
    console.log(`>>> Will try to read file ${file} into dictionary.`);
    var contents = '';
    try {
        contents = fs.readFileSync(file, { encoding: 'utf8' });
    } catch(error) {
        console.log(`>>> Failed reading from ${file}. Error ${error}.`);
    }
    contents.split(/\r|\n/).forEach(element => handleRaw(dict, element, false));
    commands.count(dict);
}

function handleDump(dict, file) {
    console.log(`>>> Will try to dump to ${file}.`);
    try {
        fs.writeFileSync(file, [...dict.values()].sort((a, b) => a.localeCompare(b, 'lt')).join('\n'));
        console.log(`>>> Write done.`);
    } catch(error) {
        console.log(`>>> Failed dumping to ${file}. Error ${error}.`);
    }
}

function handleClear(dict) {
    console.log(`>>> Clearing dictionary, but will create a backup called backupBeforeClear.dict`);
    commands.dump(dictionary, 'backupBeforeClear.dict');
    dictionary.clear();
}

function handleCount(dict) {
    console.log(`>>> Dictionary size is ${dict.size}.`);
}

function handleExit(dict) {
    console.log(`>>> Exiting, but will create a backup called backupBeforeExit.dict`);
    commands.dump(dict, 'backupBeforeExit.dict');
    process.exit();
}

console.log(`>>> Ready.`);
