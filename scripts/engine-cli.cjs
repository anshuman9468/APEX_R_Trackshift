'use strict';
require('../dist/apex-model.js');
const E = require('../dist/engine.js');
const T = require('../dist/telemetry.js');
let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {buffer += chunk;if(buffer.length > 2 * 1024 * 1024){process.stderr.write('Input too large');process.exit(1);}});
process.stdin.on('end', () => {
  try {
    const message = JSON.parse(buffer);
    let value;
    if(message.method === 'compare') value = E.compare(message.input);
    else if(message.method === 'validate') value = E.validate(message.input);
    else if(message.method === 'normalise') value = E.normalise(message.input);
    else if(message.method === 'meta') value = {version:E.VERSION,config:E.CONFIG,scenarios:E.SCENARIOS};
    else if(message.method === 'telemetry') {const state=E.normalise({state:{soc:message.input?.soc??42}}).state;value=T.synthetic(state.soc);}
    else throw new Error('Unknown engine method');
    process.stdout.write(JSON.stringify(value));
  } catch(error) {process.stderr.write(error.message);process.exitCode=1;}
});
