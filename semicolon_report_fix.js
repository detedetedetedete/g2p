function fix(input) {
  let result = '';
  for(let record of input.split('\n')) {
    let merge = false;
    for(let token of record.split(';')) {
      if(!token) {
        result += '\n';
        break;
      }
      if(token === '[S]') {
        merge = true;
      }
      if(token === '[E]') {
        merge = false;
      }
      result += token + (merge ? ':' : ';');
    }
  }
  return result;
 }
