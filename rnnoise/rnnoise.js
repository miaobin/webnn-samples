'use strict';

import {buildConstantByNpy} from '../common/utils.js';

export class RNNoise {
  constructor(modelPath, batchSize, frames) {
    this.baseUrl_ = modelPath;
    this.batchSize_ = batchSize;
    this.frames_ = frames;
    this.model_ = null;
    this.context_ = null;
    this.graph_ = null;
    this.builder_ = null;
    this.featureSize = 42;
    this.vadGruHiddenSize = 24;
    this.vadGruNumDirections = 1;
    this.noiseGruHiddenSize = 48;
    this.noiseGruNumDirections = 1;
    this.denoiseGruHiddenSize = 96;
    this.denoiseGruNumDirections = 1;
  }

  buildGruCell_(builder, input, weight, recurrentWeight, hiddenState, hiddenSize, options = {}) {
    const desc = { dataType: input.dataType(), dimensions: [1] };
    const constantBuffer0 = new Float32Array([1]).fill(0);
    const constantBuffer1 = new Float32Array([1]).fill(1);
    const zero = builder.constant(desc, constantBuffer0);
    const one = builder.constant(desc, constantBuffer1);

    const inputShape = input.shape();
    const inputSize = inputShape[1];

    // update gate (z)
    let z = builder.sigmoid(builder.add(
      builder.add(
        (options.bias ? builder.slice(options.bias, [0], [hiddenSize]) : zero),
        (options.recurrentBias ?
          builder.slice(options.recurrentBias, [0], [hiddenSize]) :
          zero)),
      builder.add(
        builder.matmul(
          input,
          builder.transpose(
            builder.slice(weight, [0, 0], [hiddenSize, inputSize]))),
        builder.matmul(
          hiddenState,
          builder.transpose(
            builder.slice(recurrentWeight, [0, 0], [hiddenSize, hiddenSize]))))));

    // reset gate (r)
    let r = builder.sigmoid(builder.add(
      builder.add(
        (options.bias ? builder.slice(options.bias, [hiddenSize], [hiddenSize]) :
          zero),
        (options.recurrentBias ?
          builder.slice(options.recurrentBias, [hiddenSize], [hiddenSize]) :
          zero)),
      builder.add(
        builder.matmul(
          input,
          builder.transpose(
            builder.slice(weight, [hiddenSize, 0], [hiddenSize, inputSize]))),
        builder.matmul(
          hiddenState,
          builder.transpose(builder.slice(
            recurrentWeight, [hiddenSize, 0], [hiddenSize, hiddenSize]))))));

    // new gate (n)
    let n;
    if (options.resetAfter) {
      n = builder.tanh(builder.add(
        (options.bias ?
          builder.slice(options.bias, [2 * hiddenSize], [hiddenSize]) :
          zero),
        builder.add(
          builder.matmul(
            input,
            builder.transpose(
              builder.slice(weight, [2 * hiddenSize, 0], [hiddenSize, inputSize]))),
          builder.mul(
            r,
            builder.add(
              (options.recurrentBias ?
                builder.slice(
                  options.recurrentBias, [2 * hiddenSize], [hiddenSize]) :
                zero),
              builder.matmul(
                hiddenState,
                builder.transpose(builder.slice(
                  recurrentWeight,
                  [2 * hiddenSize, 0],
                  [hiddenSize, hiddenSize]))))))));
    } else {
      n = builder.tanh(builder.add(
        builder.add(
          (options.bias ?
            builder.slice(options.bias, [2 * hiddenSize], [hiddenSize]) :
            zero),
          (options.recurrentBias ?
            builder.slice(options.recurrentBias, [2 * hiddenSize], [hiddenSize]) :
            zero)),
        builder.add(
          builder.matmul(
            input,
            builder.transpose(
              builder.slice(weight, [2 * hiddenSize, 0], [hiddenSize, inputSize]))),
          builder.matmul(
            builder.mul(r, hiddenState),
            builder.transpose(builder.slice(
              recurrentWeight, [2 * hiddenSize, 0], [hiddenSize, hiddenSize]))))));
    }

    // compute the new hidden state
    return builder.add(builder.mul(z, hiddenState), builder.mul(n, builder.sub(one, z)));
  }

  buildGru_(builder, input, weight, recurrentWeight, steps, hiddenSize, options = {}) {

    function squeeze(builder, op) {
      let shape = op.shape();
      let newShape = shape.slice(1);
      return builder.reshape(op, newShape);
    }

    const inputShape = input.shape();
    const batchSize = inputShape[1];
    const inputSize = inputShape[2];

    const numDirections = (options.direction == 'both' ? 2 : 1);
    let hiddenState = options.initialHiddenState;

    if (!hiddenState) {
      const desc = {
        dataType: 'float32',
        dimensions: [numDirections, 1, hiddenSize]
      };
      const totalSize = numDirections * hiddenSize;
      hiddenState = builder.constant(desc, new Float32Array(totalSize).fill(0));
    }

    let sequence = null;
    let currentWeight = [];
    let currentRecurrentWeight = [];
    let currentBias = [];
    let currentRecurrentBias = [];

    for (let dir = 0; dir < numDirections; ++dir) {
      currentWeight.push(squeeze(
        builder,
        builder.slice(weight, [dir, 0, 0], [1, 3 * hiddenSize, inputSize])));
      currentRecurrentWeight.push(squeeze(
        builder,
        builder.slice(
          recurrentWeight, [dir, 0, 0], [1, 3 * hiddenSize, hiddenSize])));
      currentBias.push(
        options.bias ?
          (squeeze(
            builder, builder.slice(options.bias, [dir, 0], [1, 3 * hiddenSize]))) :
          null);
      currentRecurrentBias.push(
        options.recurrentBias ?
          (squeeze(
            builder,
            builder.slice(options.recurrentBias, [dir, 0], [1, 3 * hiddenSize]))) :
          null);
    }

    for (let step = 0; step < steps; ++step) {
      let currentHidden = [];
      let currentOutput = null;

      for (let dir = 0; dir < numDirections; ++dir) {
        currentHidden.push(squeeze(
          builder,
          builder.slice(hiddenState, [dir, 0, 0], [1, batchSize, hiddenSize])));
      }

      for (let dir = 0; dir < numDirections; ++dir) {
        let slice =
          (dir == 1 || options.direction == 'backward' ? steps - step - 1 : step);
        let currentInput = squeeze(
          builder, builder.slice(input, [slice, 0, 0], [1, batchSize, inputSize]));

        let result = builder.reshape(
          this.buildGruCell_(
            builder,
            currentInput,
            currentWeight[dir],
            currentRecurrentWeight[dir],
            currentHidden[dir],
            hiddenSize,
            {
              bias: currentBias[dir],
              recurrentBias: currentRecurrentBias[dir],
              resetAfter: options.resetAfter,
              layout: options.layout,
              activations: options.activations
            }),
          [1, batchSize, hiddenSize]);

        currentOutput =
          (currentOutput ? builder.concat([currentOutput, result], 0) : result);
      }

      hiddenState = currentOutput;

      if (options.returnSequence) {
        currentOutput =
          builder.reshape(currentOutput, [1, numDirections, batchSize, hiddenSize]);
        sequence =
          (sequence ? builder.concat([sequence, currentOutput], 0) : currentOutput);
      }
    }

    return (sequence ? [hiddenState, sequence] : [hiddenState]);
  }

  async load(contextOptions) {
    this.context_ = await navigator.ml.createContext(contextOptions);
    this.builder_ = new MLGraphBuilder(this.context_);
    // Create constants by loading pre-trained data from .npy files.
    const inputDenseKernel0 = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'input_dense_kernel_0.npy');
    const inputDenseBias0 = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'input_dense_bias_0.npy');
    const vadGruW = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'vad_gru_W.npy');
    const vadGruR = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'vad_gru_R.npy');
    const vadGruBData = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'vad_gru_B.npy');
    const noiseGruW = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'noise_gru_W.npy');
    const noiseGruR = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'noise_gru_R.npy');
    const noiseGruBData = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'noise_gru_B.npy');
    const denoiseGruW = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_gru_W.npy');
    const denoiseGruR = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_gru_R.npy');
    const denoiseGruBData = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_gru_B.npy');
    const denoiseOutputKernel0 = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_output_kernel_0.npy');
    const denoiseOutputBias0 = await buildConstantByNpy(this.builder_,
        this.baseUrl_ + 'denoise_output_bias_0.npy');
    // Build up the network.
    const input = this.builder_.input('input', {
      dataType: 'float32',
      dimensions: [this.batchSize_, this.frames_, this.featureSize],
    });
    const inputDense0 = this.builder_.matmul(input, inputDenseKernel0);
    const biasedTensorName2 = this.builder_.add(inputDense0, inputDenseBias0);
    const inputDenseTanh0 = this.builder_.tanh(biasedTensorName2);
    const vadGruX = this.builder_.transpose(
        inputDenseTanh0, {permutation: [1, 0, 2]});
    const vadGruB = this.builder_.slice(
        vadGruBData, [0, 0], [1, 3 * this.vadGruHiddenSize]);
    const vadGruRB = this.builder_.slice(
        vadGruBData,
        [0, 3 * this.vadGruHiddenSize],
        [1, 3 * this.vadGruHiddenSize]);
    const vadGruInitialH = this.builder_.input('vadGruInitialH', {
      dataType: 'float32',
      dimensions: [1, this.batchSize_, this.vadGruHiddenSize],
    });
    const [vadGruYH, vadGruY] = this.buildGru_(this.builder_, vadGruX,
    // const [vadGruYH, vadGruY] = this.builder_.gru(vadGruX,
        vadGruW, vadGruR, this.frames_, this.vadGruHiddenSize, {
          bias: vadGruB,
          recurrentBias: vadGruRB,
          initialHiddenState: vadGruInitialH,
          returnSequence: true,
          resetAfter: false,
          activations: [this.builder_.sigmoid(), this.builder_.relu()],
        });
    const vadGruYTransposed = this.builder_.transpose(
        vadGruY, {permutation: [2, 0, 1, 3]});
    const vadGruTranspose1 = this.builder_.reshape(
        vadGruYTransposed, [1, this.frames_, this.vadGruHiddenSize]);
    const concatenate1 = this.builder_.concat(
        [inputDenseTanh0, vadGruTranspose1, input], 2);
    const noiseGruX = this.builder_.transpose(
        concatenate1, {permutation: [1, 0, 2]});
    const noiseGruB = this.builder_.slice(
        noiseGruBData, [0, 0], [1, 3 * this.noiseGruHiddenSize]);
    const noiseGruRB = this.builder_.slice(
        noiseGruBData,
        [0, 3 * this.noiseGruHiddenSize],
        [1, 3 * this.noiseGruHiddenSize]);
    const noiseGruInitialH = this.builder_.input('noiseGruInitialH', {
      dataType: 'float32',
      dimensions: [1, this.batchSize_, this.noiseGruHiddenSize],
    });
    const [noiseGruYH, noiseGruY] = this.buildGru_(this.builder_, noiseGruX,
    // const [noiseGruYH, noiseGruY] = this.builder_.gru(noiseGruX,
        noiseGruW, noiseGruR, this.frames_, this.noiseGruHiddenSize, {
          bias: noiseGruB,
          recurrentBias: noiseGruRB,
          initialHiddenState: noiseGruInitialH,
          returnSequence: true,
          resetAfter: false,
          activations: [this.builder_.sigmoid(), this.builder_.relu()],
        });
    const noiseGruYTransposed = this.builder_.transpose(
        noiseGruY, {permutation: [2, 0, 1, 3]});
    const noiseGruTranspose1 = this.builder_.reshape(
        noiseGruYTransposed, [1, this.frames_, this.noiseGruHiddenSize]);
    const concatenate2 = this.builder_.concat(
        [vadGruTranspose1, noiseGruTranspose1, input], 2);
    const denoiseGruX = this.builder_.transpose(
        concatenate2, {permutation: [1, 0, 2]});
    const denoiseGruB = this.builder_.slice(
        denoiseGruBData, [0, 0], [1, 3 * this.denoiseGruHiddenSize]);
    const denoiseGruRB = this.builder_.slice(
        denoiseGruBData,
        [0, 3 * this.denoiseGruHiddenSize],
        [1, 3 * this.denoiseGruHiddenSize]);
    const denoiseGruInitialH = this.builder_.input('denoiseGruInitialH', {
      dataType: 'float32',
      dimensions: [1, this.batchSize_, this.denoiseGruHiddenSize],
    });
    const [denoiseGruYH, denoiseGruY] = this.buildGru_(this.builder_, denoiseGruX,
    // const [denoiseGruYH, denoiseGruY] = this.builder_.gru(denoiseGruX,
        denoiseGruW, denoiseGruR, this.frames_, this.denoiseGruHiddenSize, {
          bias: denoiseGruB,
          recurrentBias: denoiseGruRB,
          initialHiddenState: denoiseGruInitialH,
          returnSequence: true,
          resetAfter: false,
          activations: [this.builder_.sigmoid(), this.builder_.relu()],
        });
    const denoiseGruYTransposed = this.builder_.transpose(
        denoiseGruY, {permutation: [2, 0, 1, 3]});
    const denoiseGruTranspose1 = this.builder_.reshape(
        denoiseGruYTransposed, [1, this.frames_, this.denoiseGruHiddenSize]);
    const denoiseOutput0 = this.builder_.matmul(
        denoiseGruTranspose1, denoiseOutputKernel0);
    const biasedTensorName = this.builder_.add(
        denoiseOutput0, denoiseOutputBias0);
    const denoiseOutput = this.builder_.sigmoid(biasedTensorName);

    return {denoiseOutput, vadGruYH, noiseGruYH, denoiseGruYH};
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build(outputOperand);
  }

  async compute(inputs, outputs) {
    const results = await this.context_.compute(this.graph_, inputs, outputs);
    return results.outputs;
  }
}
