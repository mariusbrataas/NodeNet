import { DenseLayer } from './Dense.js';
import { DenseNet } from './DenseNet.js';
import { Tensor } from './Tensor.js';

const net = new DenseNet(3, 4, 'sigmoid', [
  { features: 7, act: 'tanh' },
  { features: 8, act: 'tanh' }
]);

var X = new Tensor<8, 3>([
  [0, 0, 0],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 1],
  [1, 0, 0],
  [1, 0, 1],
  [1, 1, 0],
  [1, 1, 1]
]);

var Y = new Tensor<8, 3>(
  X.value
    .map(row => row.reduce((tot, val) => tot + val, 0))
    .map(val =>
      [val === 1, val === 2, val === 3, val % 3].map(v => (v ? 1 : 0))
    )
);

net.fit(X, Y, 1500, { 0: 0.001, 100: 0.01, 500: 0.1 });

const log = (msg: string, ...args: any[]) => console.log(`\n${msg}`, ...args);

net.print();

log('Features: ', X.value);
log('Prediction: ', net.state!.value);
log('Rounded: ', net.state!.round().value);
log('Target: ', Y.value);
