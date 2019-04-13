const fs = require('fs');

let dir = fs.readdirSync('.');

dir = dir.filter(d => d.match(/[0-9]{5,6}$/)).filter(d => fs.existsSync(`./${d}/summary.log`));
let summaries = dir.map(d => ({ dir: d, summary: `./${d}/summary.log`}));

for(let summary of summaries) {
  let content = fs.readFileSync(summary.summary, { encoding: 'utf8' });
  let validation = content.match(/-Validation set-*[^-]*?Had 0 mistakes: [0-9]* \(([0-9.]*)%\)/);
  validation = validation ? validation[1] : 0;
  let full = content.match(/-Full set-*[^-]*?Had 0 mistakes: [0-9]* \(([0-9.]*)%\)/);
  full = full ? full[1] : 0;
  fs.renameSync(summary.dir, `${summary.dir}-${validation}-${full}`);
}

