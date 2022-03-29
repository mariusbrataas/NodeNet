import { Layer } from './Layer.js';

export const generateId = (() => {
  var count = 0;
  return () => `l_${(count++).toString(36)}`;
})();

export const printLayers = (layer: Layer<number, number>) => {
  const layers = layer.listLayers();

  var columnWidths: { [column: string]: number } = { type: 4 };

  layers
    .map(layer => {
      columnWidths['type'] = Math.max(
        columnWidths['type'],
        layer.constructor.name.length
      );
      return layer.getInfo();
    })
    .forEach(layer =>
      Object.keys(layer).forEach(
        column =>
          (columnWidths[column] = Math.max(
            columnWidths[column] || 0,
            column.length,
            `${layer[column]}`.length
          ))
      )
    );

  const colsOrder = [
    'type',
    ...Object.keys(columnWidths)
      .filter(key => key !== 'type')
      .sort()
  ];

  const rjust = (str: string, minLength: number) => {
    var output = `${str}`;
    while (output.length < minLength) output += ' ';
    return output;
  };

  console.log(
    [
      '',
      colsOrder.map(col => rjust(`${col}`, columnWidths[col]) + '  ').join(''),
      ...layers.map(layer => {
        const info = {
          type: layer.constructor.name,
          ...layer.getInfo()
        } as any;
        return colsOrder
          .map(col => rjust(`${info[col] ?? ''}`, columnWidths[col]) + '  ')
          .join('');
      })
    ].join('\n')
  );
};
