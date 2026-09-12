importScripts('./engine.js');
self.onmessage = function (event) {
  try {
    if (event.data.type !== 'validate') throw new Error('Unknown worker operation');
    self.postMessage({result: self.ApexEngine.validate(event.data.input)});
  } catch (error) { self.postMessage({error: error.message}); }
};
