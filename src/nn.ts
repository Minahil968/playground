// Neural Network Module

// Define Activation Functions
const activationFunctions = {
  sigmoid: {
    output: (x) => 1 / (1 + Math.exp(-x)),
    derivative: (x) => x * (1 - x),
  },
  relu: {
    output: (x) => Math.max(0, x),
    derivative: (x) => x <= 0 ? 0 : 1,
  },
  tanh: {
    output: (x) => Math.tanh(x),
    derivative: (x) => 1 - Math.pow(Math.tanh(x), 2),
  },
};

// Define Error Functions
const errorFunctions = {
  meanSquaredError: {
    error: (output, target) => 0.5 * Math.pow(output - target, 2),
    derivative: (output, target) => output - target,
  },
};

// Define Regularization Functions
const regularizationFunctions = {
  l1: {
    output: (weight) => Math.abs(weight),
    derivative: (weight) => weight < 0 ? -1 : (weight > 0 ? 1 : 0),
  },
  l2: {
    output: (weight) => 0.5 * Math.pow(weight, 2),
    derivative: (weight) => weight,
  },
};

// Node Class
class Node {
  constructor(id, activationFunction, bias = 0.1) {
    this.id = id;
    this.activationFunction = activationFunction;
    this.bias = bias;
    this.inputLinks = [];
    this.outputLinks = [];
    this.totalInput = 0;
    this.output = 0;
    this.outputDerivative = 0;
    this.inputDerivative = 0;
    this.accumulatedInputDerivative = 0;
    this.numAccumulatedDerivatives = 0;
  }

  updateOutput() {
    this.totalInput = this.bias;
    for (const link of this.inputLinks) {
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activationFunction.output(this.totalInput);
    return this.output;
  }
}

// Link Class
class Link {
  constructor(source, destination, regularizationFunction, weight = Math.random() - 0.5) {
    this.id = `${source.id}-${destination.id}`;
    this.source = source;
    this.destination = destination;
    this.weight = weight;
    this.isDead = false;
    this.errorDerivative = 0;
    this.accumulatedErrorDerivative = 0;
    this.numAccumulatedDerivatives = 0;
    this.regularizationFunction = regularizationFunction;
  }
}

// Neural Network Class
class NeuralNetwork {
  constructor(networkShape, activationFunction, outputActivationFunction, regularizationFunction) {
    this.layers = [];
    for (let i = 0; i < networkShape.length; i++) {
      const layer = [];
      for (let j = 0; j < networkShape[i]; j++) {
        const node = new Node(`${i}-${j}`, i === networkShape.length - 1 ? outputActivationFunction : activationFunction);
        layer.push(node);
      }
      this.layers.push(layer);
    }
    for (let i = 0; i < this.layers.length - 1; i++) {
      for (const source of this.layers[i]) {
        for (const destination of this.layers[i + 1]) {
          const link = new Link(source, destination, regularizationFunction);
          source.outputLinks.push(link);
          destination.inputLinks.push(link);
        }
      }
    }
  }

  forwardPropagation(inputs) {
    for (let i = 0; i < inputs.length; i++) {
      this.layers[0][i].output = inputs[i];
    }
    for (let i = 1; i < this.layers.length; i++) {
      for (const node of this.layers[i]) {
        node.updateOutput();
      }
    }
    return this.layers[this.layers.length - 1][0].output;
  }

  backPropagation(target, errorFunction) {
    const outputNode = this.layers[this.layers.length - 1][0];
    outputNode.outputDerivative = errorFunction.derivative(outputNode.output, target);
    for (let i = this.layers.length - 1; i >= 1; i--) {
      for (const node of this.layers[i]) {
        node.inputDerivative = node.outputDerivative * node.activationFunction.derivative(node.totalInput);
        node.accumulatedInputDerivative += node.inputDerivative;
        node.numAccumulatedDerivatives++;
        for (const link of node.inputLinks) {
          link.errorDerivative = node.inputDerivative * link.source.output;
          link.accumulatedError
