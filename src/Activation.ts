function addLimit(
  func: (x: number) => number,
  low: number = -Infinity,
  high: number = Infinity
) {
  return (x: number) => {
    var output = func(x);
    if (output > high) {
      output = high;
    } else if (output < low) {
      output = low;
    }
    return output;
  };
}

export interface Activation {
  title: string;
  forward: (x: number) => number;
  backward: (x: number) => number;
}

const Sigmoid: Activation = {
  title: 'Sigmoid',
  forward: addLimit((x: number) => 1 / (1 + Math.exp(-x)), 0, 1),
  backward: addLimit(
    (x: number) => {
      const fwd = Sigmoid.forward(x);
      return fwd * (1 - fwd);
    },
    0,
    1
  )
};

const Tanh: Activation = {
  title: 'Tanh',
  forward: addLimit((x: number) => Math.tanh(x), -1, 1),
  backward: addLimit((x: number) => 1 / Math.cosh(x) ** 2, 0, 1)
};

export const getActivation = (() => {
  const activations = { sigmoid: Sigmoid, tanh: Tanh };
  return (name: keyof typeof activations) => activations[name];
})();

export type Activations = Parameters<typeof getActivation>[0];
