import { Tensor } from './Tensor.js';

type ShiftDouble<Arr extends any[]> = Arr extends [
  arg0: any,
  arg1: any,
  ...rest: infer U
]
  ? U
  : Arr;

export abstract class Layer<
  FeaturesIn extends number,
  FeaturesOut extends number,
  CurrentStates extends { [key: string]: Tensor<number, number> } = {}
> {
  protected prevLayer?: Layer<number, FeaturesIn>;
  protected nextLayer?: Layer<FeaturesOut, number>;

  public readonly featuresIn: FeaturesIn;
  public readonly featuresOut: FeaturesOut;

  public current: {
    input?: Tensor<number, FeaturesIn>;
    output?: Tensor<number, FeaturesOut>;
  } & Partial<CurrentStates> = {};

  public past: Layer<FeaturesIn, FeaturesOut, CurrentStates>['current'][] = [];

  constructor(
    featuresIn: FeaturesIn | Layer<number, FeaturesIn>,
    featuresOut: FeaturesOut | Layer<FeaturesOut, number>
  ) {
    if (typeof featuresIn === 'number') {
      this.featuresIn = featuresIn;
    } else {
      this.featuresIn = featuresIn.featuresOut;
      this.prevLayer = featuresIn;
      this.prevLayer.nextLayer = this;
    }
    if (typeof featuresOut === 'number') {
      this.featuresOut = featuresOut;
    } else {
      this.featuresOut = featuresOut.featuresIn;
      this.nextLayer = featuresOut;
      this.nextLayer.prevLayer = this;
    }
  }

  public get state() {
    return this.current?.output;
  }

  public get layerType() {
    return this.constructor.name;
  }

  public get firstLayer(): Layer<number, number> {
    return this.prevLayer?.firstLayer || this;
  }

  public get lastLayer(): Layer<number, number> {
    return this.nextLayer?.lastLayer || this;
  }

  public addLayer = <
    Features extends number,
    T extends new (
      featuresIn: number | Layer<number, number>,
      featuresOut: Features
    ) => Layer<number, Features>
  >(
    layerConstructor: T,
    features: Features,
    ...args: ShiftDouble<ConstructorParameters<T>>
  ) =>
    new layerConstructor(
      this.lastLayer,
      features,
      //@ts-ignore
      ...args
    );

  public listLayers = (startAtFirst = true): Layer<number, number>[] =>
    startAtFirst
      ? this.firstLayer.listLayers(false)
      : [this, ...(this.nextLayer ? this.nextLayer.listLayers(false) : [])];

  /**
   * Save current state in past values
   */
  public remember = () => {
    this.past.push(this.current);
  };

  /**
   * Load state from past values
   */
  public recall = () => {
    this.current = this.past.pop() || {};
  };

  public forwardSequential = <N extends number>(
    features: Tensor<N, FeaturesIn>
  ) => {
    this.remember();
    return this.forward(features);
  };

  public backwardSequential = <N extends number>(
    error: Tensor<N, FeaturesOut>
  ) => {
    const out = this.backward(error);
    this.recall();
    return out;
  };

  private recForward = <N extends number>(
    features: Tensor<N, FeaturesIn>
  ): Tensor<N, number> => {
    const output = this.forward(features);
    return this.nextLayer ? this.nextLayer.recForward(output) : output;
  };

  public propagate = <N extends number>(features: Tensor<N, FeaturesIn>) =>
    this.firstLayer.recForward(features);

  private recBackward = <N extends number>(
    error: Tensor<N, FeaturesOut>
  ): Tensor<N, number> => {
    const prevLayerError = this.backward(error);
    return this.prevLayer
      ? this.prevLayer.recBackward(prevLayerError)
      : prevLayerError;
  };

  public backpropagate = <N extends number>(error: Tensor<N, number>) =>
    this.lastLayer.recBackward(error);

  abstract getInfo(): { [key: string]: string | number | boolean };

  abstract forward<N extends number>(
    features: Tensor<N, FeaturesIn>
  ): Tensor<N, FeaturesOut>;

  abstract backward<N extends number>(
    error: Tensor<N, FeaturesOut>
  ): Tensor<N, FeaturesIn>;

  abstract updateWeights(lr?: number): void;
}
