const transpose = (arr: number[][]) =>
  arr[0].map((_, colIdx) => arr.map(row => row[colIdx]));

const initMatrix = <
  D0 extends number = 0,
  D1 extends number = 0,
  D2 extends number = 0
>(
  initFunc: (() => number) | number,
  d0?: D0,
  d1?: D1,
  d2?: D2
): D0 extends 0
  ? number
  : D1 extends 0
  ? number[]
  : D2 extends 0
  ? number[][]
  : number[][][] => {
  if (!d0)
    return (typeof initFunc === 'function' ? initFunc() : initFunc) as any;
  return Array(d0)
    .fill(undefined)
    .map(() => initMatrix(initFunc, d1, d2)) as any;
};

export class Tensor<D0 extends number, D1 extends number> {
  public static applyInplace = <
    D0 extends number,
    D1 extends number,
    T extends Tensor<D0, D1> | Tensor<1, D1> | number | undefined = undefined
  >(
    operator: (value0: number, ...values: (number | undefined)[]) => number,
    t0: Tensor<D0, D1>,
    ...tensors: T[]
  ): Tensor<D0, D1> => {
    t0.value.forEach((row, rowIdx) =>
      row.forEach(
        (col, colIdx) =>
          (row[colIdx] = operator(
            col,
            ...tensors.map(
              tensor =>
                (tensor instanceof Tensor
                  ? (tensor.value[rowIdx] || tensor.value[0])[colIdx]
                  : tensor) as number
            )
          ))
      )
    );
    return t0;
  };

  public static apply = <
    D0 extends number,
    D1 extends number,
    T extends Tensor<D0, D1> | Tensor<1, D1> | number | undefined = undefined
  >(
    operator: (value0: number, ...values: (number | undefined)[]) => number,
    t0: Tensor<D0, D1>,
    ...tensors: T[]
  ): Tensor<D0, D1> =>
    new Tensor(
      t0.value.map((row, rowIdx) =>
        row.map((col, colIdx) =>
          operator(
            col,
            ...tensors.map(
              tensor =>
                (tensor instanceof Tensor
                  ? (tensor.value[rowIdx] || tensor.value[0])[colIdx]
                  : tensor) as number
            )
          )
        )
      )
    );

  public static sum = <D0 extends number, D1 extends number>(
    t0: Tensor<D0, D1>,
    ...tensors: (Tensor<D0, D1> | Tensor<1, D1> | number | undefined)[]
  ) =>
    Tensor.apply(
      (v0: number, ...values) =>
        values.reduce((tot = 0, val = 1) => tot + val, v0) as number,
      t0,
      ...tensors
    );

  public static empty = <D0 extends number, D1 extends number>(
    initFunc: (() => number) | number,
    d0: D0,
    d1: D1
  ) => new Tensor<D0, D1>(initMatrix(initFunc, d0, d1));

  public static random = <D0 extends number, D1 extends number>(
    d0: D0,
    d1: D1
  ) => Tensor.empty(() => Math.random() * 2 - 1, d0, d1);

  public static randomLike = <D0 extends number, D1 extends number>(
    tensor: Tensor<D0, D1>
  ): Tensor<D0, D1> =>
    Tensor.random(tensor.value.length, tensor.value[0].length);

  public static zeros = <D0 extends number, D1 extends number>(
    d0: D0,
    d1: D1
  ) => Tensor.empty(0, d0, d1);

  public static zerosLike = <D0 extends number, D1 extends number>(
    tensor: Tensor<D0, D1>
  ): Tensor<D0, D1> =>
    Tensor.zeros(tensor.value.length, tensor.value[0].length);

  public static ones = <D0 extends number, D1 extends number>(d0: D0, d1: D1) =>
    Tensor.empty(1, d0, d1);

  public static onesLike = <D0 extends number, D1 extends number>(
    tensor: Tensor<D0, D1>
  ): Tensor<D0, D1> => Tensor.ones(tensor.value.length, tensor.value[0].length);

  constructor(public value: number[][]) {}

  public get T() {
    return new Tensor<D1, D0>(transpose(this.value));
  }

  public sum = () => {
    var total = 0;
    this.value.forEach(row => row.forEach(col => (total += col)));
    return total;
  };

  public sumRows = () =>
    new Tensor<1, D1>([
      this.value.map(row => row.reduce((total, col) => total + col, 0))
    ]);

  public sumColumns = () =>
    new Tensor<D0, 1>(
      this.value[0].map((_, colIdx) => [
        this.value.reduce((total, row) => total + row[colIdx], 0)
      ])
    );

  public mean = () => {
    var total = 0;
    var count = 0;
    this.value.forEach(row =>
      row.forEach(col => {
        total += col;
        count += 1;
      })
    );
    return total / count;
  };

  public meanRows = () =>
    new Tensor<1, D1>([
      this.value.map(
        row => row.reduce((total, col) => total + col, 0) / row.length
      )
    ]);

  public meanColumns = () =>
    new Tensor<D0, 1>(
      this.value[0].map((_, colIdx) => [
        this.value.reduce((total, row) => total + row[colIdx], 0) /
          this.value.length
      ])
    );

  public dot = <D extends number>(tensor: Tensor<D1, D>) => {
    const m1 = this.value;
    const m2 = tensor.T.value;
    return new Tensor<D0, D>(
      m1.map(a =>
        m2.map(b => a.reduce((tot, val, idx) => tot + val * b[idx], 0))
      )
    );
  };

  public round = (inplace?: boolean) =>
    (inplace ? Tensor.applyInplace : Tensor.apply)(a => Math.round(a), this);

  public abs = (inplace?: boolean) =>
    (inplace ? Tensor.applyInplace : Tensor.apply)(a => Math.abs(a), this);

  public square = (inplace?: boolean) =>
    (inplace ? Tensor.applyInplace : Tensor.apply)(a => a ** 2, this);

  public add = (
    tensor: Tensor<D0, D1> | Tensor<1, D1> | number,
    inplace?: boolean
  ) =>
    (inplace ? Tensor.applyInplace : Tensor.apply)(
      (a, b) => a + b!,
      this,
      tensor
    );

  public subtract = (
    tensor: Tensor<D0, D1> | Tensor<1, D1> | number,
    inplace?: boolean
  ) =>
    (inplace ? Tensor.applyInplace : Tensor.apply)(
      (a, b) => a - b!,
      this,
      tensor
    );

  public multiply = (
    tensor: Tensor<D0, D1> | Tensor<1, D1> | number,
    inplace?: boolean
  ) =>
    (inplace ? Tensor.applyInplace : Tensor.apply)(
      (a, b) => a * b!,
      this,
      tensor
    );

  public divide = (
    tensor: Tensor<D0, D1> | Tensor<1, D1> | number,
    inplace?: boolean
  ) =>
    (inplace ? Tensor.applyInplace : Tensor.apply)(
      (a, b) => a / b!,
      this,
      tensor
    );
}
