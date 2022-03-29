import { Activation, Activations, getActivation } from './Activation.js';
import { Layer } from './Layer.js';
import { Tensor } from './Tensor.js';

export class DenseLayer<
  FeaturesIn extends number,
  FeaturesOut extends number
> extends Layer<
  FeaturesIn,
  FeaturesOut,
  { weightedInput: Tensor<number, FeaturesOut> }
> {
  private activation: Activation;

  private readonly weights: Tensor<FeaturesIn, FeaturesOut> = Tensor.random(
    this.featuresIn,
    this.featuresOut
  );
  private readonly bias: Tensor<1, FeaturesOut> = Tensor.random(
    1,
    this.featuresOut
  );

  private gradients: {
    weights?: Tensor<FeaturesIn, FeaturesOut>;
    bias?: Tensor<1, FeaturesOut>;
  } = {};

  constructor(
    featuresIn: FeaturesIn | Layer<number, FeaturesIn>,
    featuresOut: FeaturesOut | Layer<FeaturesOut, number>,
    activation: Activations = 'sigmoid'
  ) {
    super(featuresIn, featuresOut);
    this.activation = getActivation(activation);
  }

  public getInfo = () => ({
    parameters: (this.featuresIn + 1) * this.featuresOut,
    activation: this.activation.title
  });

  public forward = <N extends number>(
    features: Tensor<N, FeaturesIn> = this.prevLayer?.current.output!
  ): Tensor<N, FeaturesOut> => {
    this.current.input = features;
    this.current.weightedInput = features
      .dot(this.weights)
      .add(this.bias, true);
    this.current.output = Tensor.apply(
      this.activation.forward,
      this.current.weightedInput
    );
    return this.current.output;
  };

  public backward = <N extends number>(
    error: Tensor<N, FeaturesOut>
  ): Tensor<N, FeaturesIn> => {
    const delta = error.multiply(
      Tensor.apply(this.activation.backward, this.current.weightedInput!)
    );
    this.calculateGradients(delta);
    return delta.dot(this.weights.T);
  };

  public calculateGradients = <N extends number>(
    delta: Tensor<N, FeaturesOut>
  ) => {
    const gradientWeights = this.current.input!.T.dot(delta);
    if (this.gradients.weights) {
      this.gradients.weights.add(gradientWeights, true);
    } else {
      this.gradients.weights = gradientWeights;
    }
    const gradientBias = delta.meanRows();
    if (this.gradients.bias) {
      this.gradients.bias.add(gradientBias, true);
    } else {
      this.gradients.bias = gradientBias;
    }
  };

  public updateWeights = (lr = 1) => {
    this.weights.subtract(this.gradients.weights!.multiply(lr, true), true);
    this.gradients.weights!.multiply(0, true);
    this.bias.subtract(this.gradients.bias!.multiply(lr, true), true);
    this.gradients.bias!.multiply(0, true);
  };
}
