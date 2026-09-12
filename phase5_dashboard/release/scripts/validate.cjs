'use strict';
const fs=require('node:fs');
const path=require('node:path');
const E=require('../dist/engine.js');
const result=E.validate({count:60,seed:2026});
const dir=path.join(__dirname,'../runtime');fs.mkdirSync(dir,{recursive:true});
fs.writeFileSync(path.join(dir,'benchmark.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify({count:result.count,wins:result.wins,ties:result.ties,losses:result.losses,rollouts:result.comparisons,elapsedMs:result.elapsedMs,policies:result.rows},null,2));
