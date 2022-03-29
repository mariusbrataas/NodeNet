import { Activations } from './Activation.js';
import { DenseLayer } from './Dense.js';
import { Layer } from './Layer.js';
import { printLayers } from './Layer.utils.js';
import { Tensor } from './Tensor.js';

export class DenseNet<FeaturesIn extends number, FeaturesOut extends number> {
  public readonly layers: Layer<number, number>[];

  constructor(
    public readonly featuresIn: FeaturesIn,
    public readonly featuresOut: FeaturesOut,
    activation: Activations = 'sigmoid',
    hiddenLayers?: { features: number; act?: Activations }[]
  ) {
    this.layers = (
      [
        ...(hiddenLayers || []).slice(1),
        { features: featuresOut, act: activation }
      ].reduce(
        (prevLayer, { features, act }) =>
          new DenseLayer(prevLayer, features, act),
        new DenseLayer(
          featuresIn,
          hiddenLayers?.[0]?.features || featuresOut,
          hiddenLayers?.[0]?.act || activation
        ) as DenseLayer<number, number>
      ) as DenseLayer<number, FeaturesOut>
    ).listLayers();
  }

  public get state() {
    return this.layers[this.layers.length - 1].state;
  }

  public print = () => printLayers(this.layers[0]);

  public forward = <N extends number>(
    state: Tensor<N, FeaturesIn>
  ): Tensor<N, FeaturesOut> =>
    this.layers.reduce(
      (prevState, layer) => layer.forward(prevState),
      state as Tensor<N, number>
    );

  public backward = <N extends number>(
    error: Tensor<N, FeaturesOut>
  ): Tensor<N, FeaturesIn> =>
    this.layers.reduceRight(
      (nextError, layer) => layer.backward(nextError),
      error as Tensor<N, number>
    );

  public fullPass = <N extends number>(
    state: Tensor<N, FeaturesIn>,
    target: Tensor<N, FeaturesOut>
  ) => {
    const predictions = this.forward(state);
    const error = predictions.subtract(target);
    this.backward(error);
    return error;
  };

  public updateWeights = (lr = 1) =>
    this.layers.forEach(layer => layer.updateWeights(lr));

  public fit = <N extends number>(
    state: Tensor<N, FeaturesIn>,
    target: Tensor<N, FeaturesOut>,
    epochs = 1000,
    lr: number | { [epoch: number]: number } = 0.01
  ) => {
    const printStep = Math.round(epochs / 20);
    const getLearningRate =
      typeof lr === 'number'
        ? () => lr
        : (() => {
            var prevLearningRate = lr[0] || 0.01;
            return (epoch: number) =>
              (prevLearningRate = lr[epoch] ?? prevLearningRate);
          })();
    for (var epoch = 0; epoch < epochs; epoch++) {
      const error = this.fullPass(state, target).square().mean();
      this.updateWeights(getLearningRate(epoch));
      if (epoch % printStep === 0) console.log(`Epoch ${epoch}: ${error}`);
    }
  };
}
