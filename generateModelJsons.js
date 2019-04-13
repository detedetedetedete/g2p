
let models = [
  'LSTM - Nadam - rnn[256] -> Dense<softmax>',
  'LSTM - Nadam - rnn[114] -> rnn[38] -> Dense<softmax>',
  'LSTM - Nadam - rnn[114] -> Dense[38]<relu> -> rnn[38] -> Dense<softmax>',
  'LSTM - Nadam - rnn[152] -> rnn[76] -> Dense<softmax>',
  'LSTM - Nadam - rnn[76] -> rnn[152] -> Dense<softmax>',
  'LSTM - Nadam - rnn[152] -> rnn[76] -> Dense[76]<relu> -> Dense<softmax>',
  'GRU - Nadam - rnn[256] -> Dense<softmax>',
  'GRU - Nadam - rnn[114] -> rnn[38] -> Dense<softmax>',
  'GRU - Nadam - rnn[114] -> Dense[38]<relu> -> rnn[38] -> Dense<softmax>',
  'GRU - Nadam - rnn[152] -> rnn[76] -> Dense<softmax>',
  'GRU - Nadam - rnn[76] -> rnn[152] -> Dense<softmax>',
  'GRU - Nadam - rnn[152] -> rnn[76] -> Dense[76]<relu> -> Dense<softmax>'
];

function firstUpper(str) {
  return str[0].toUpperCase() + str.substr(1);
}

function generateModel(str) {
  let [rnn, optimizer, layers] = str.split(' - ');
  layers = layers.split(' -> ');
  
  let model_name = 'S2S';
  let layers_str = layers.reduce((acc, el) => {
    if(acc) {
      acc += ',\n    ';
    }
    let name = el.match(/^[^[<]*/)[0];
    name = name === 'rnn' ? rnn : name;
    
    let units = el.match(/\[([0-9]*)\]/);
    units = units ? units[1] : units;
    
    let activation = el.match(/<([^<>]*?)>/);
    activation = activation ? activation[1] : activation;
    
    model_name += `-${name}${units ? units : ''}${activation ? firstUpper(activation) : ''}`;
    
    let params = '';
    if(units && activation) {
      params = `"units":  ${units}, "activation": "${activation}"`;
    } else if(units) {
      params = `"units":  ${units}`;
    } else if(activation) {
      params = `"activation": "${activation}"`;
    }
    
    acc += `{ "type": "${name}", "params": { ${params} } }`;
    
    return acc;
  }, '');
  
  model_name += `-${optimizer}`;
  
  return { name: model_name, content: `{
  "name": "${model_name}",
  "in_tokens": [ "a", "ą", "b", "c", "č", "d", "e", "ę", "ė", "f", "g",
                    "h", "i", "į", "y", "j", "k", "l", "m", "n", "o", "p",
                    "r", "s", "š", "t", "u", "ų", "ū", "v", "z", "ž" ],
  "out_tokens": [ "A", "A_", "B", "C", "C2", "CH", "D", "DZ", "DZ2", "E", "E_",
                     "E3_", "F", "G", "H", "I", "I_", "IO_", "IU", "IU_", "J.", "K", "L",
                     "M", "N", "O_", "P", "R", "S", "S2", "T", "U", "U_", "V", "Z", "Z2" ],
  "max_in_length": 19,
  "max_out_length": 20,
  "compile": {
    "loss": "categorical_crossentropy",
    "optimizer": {
      "type": "${optimizer}",
      "args": [],
      "params": {
        "lr": 0.0005
      }
    }
  },
  "layers": [
    ${layers_str}
  ]
}`};
}

const fs = require('fs');

for(let model of models) {
  model = generateModel(model);
  fs.writeFileSync(`${model.name}.json`, model.content);
}
