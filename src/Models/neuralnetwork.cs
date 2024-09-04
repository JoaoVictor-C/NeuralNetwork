using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.src.Utils;
using NeuralNetwork.src.Layers;

namespace NeuralNetwork.src.Models
{
    public class NeuralNet
    {
        // Properties
        public List<Layer> Layers { get; set; }
        public int[] LayerSizes { get; set; }
        public Activation[] Activations { get; set; }
        public double LearningRate { get; set; }
        public int Epochs { get; set; }
        public int Epoch { get; set; }
        public int BatchSize { get; set; }
        public int Verbose { get; set; }
        public List<double> Loss { get; set; } = new List<double>();
        public List<double> Accuracy { get; set; } = new List<double>();
        public System.Random Rng { get; set; }
        public double RegularizationStrength { get; set; }
        public IOptimizer Optimizer { get; set; }
        public OptimizerType OptimizerType { get; private set; }

        // Constructor
        public NeuralNet(int[] layerSizes, Activation[] activations, double learningRate, int epochs, int batchSize, int verbose, System.Random rng, double regularizationStrength, OptimizerType optimizerType)
        {
            Layers = new List<Layer>();
            Loss = new List<double>();
            Accuracy = new List<double>();
            Rng = rng;
            LearningRate = learningRate;
            Epochs = epochs;
            Epoch = 0;
            BatchSize = batchSize;
            Verbose = verbose;
            Activations = activations;
            this.LayerSizes = layerSizes;
            Rng = new System.Random();
            OptimizerType = optimizerType;
            Optimizer = CreateOptimizer(optimizerType, learningRate, layerSizes[0] * layerSizes[1], layerSizes[1]);

            // Initialize layers with the optimizer
            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                Activation activation = activations[i];
                int inputSize = layerSizes[i];
                int outputSize = layerSizes[i + 1];
                double[] weights = new double[inputSize * outputSize];
                double[] biases = new double[outputSize];
                IOptimizer layerOptimizer = Optimizer.Clone();
                Layers.Add(new Layer(activation, inputSize, outputSize, weights, biases, learningRate, Rng, regularizationStrength, layerOptimizer));
            }
        }

        private IOptimizer CreateOptimizer(OptimizerType type, double learningRate, int weightSize, int biasSize)
        {
            switch (type)
            {
                case OptimizerType.SGD:
                    return new SGD(learningRate);
                case OptimizerType.Momentum:
                    return new Momentum(learningRate, 0.9, weightSize, biasSize);
                case OptimizerType.Adam:
                    return new Adam(learningRate, 0.9, 0.999, 1e-8, weightSize, biasSize);
                case OptimizerType.RMSprop:
                    return new RMSprop(learningRate, 0.9, 1e-8, weightSize, biasSize);
                default:
                    throw new ArgumentException("Invalid optimizer type");
            }
        }

        // Classify input with probabilities
        public double[] Classify(double[] input)
        {
            Vector<double> outputs = Vector<double>.Build.DenseOfArray(input);
            foreach (var layer in Layers)
            {
                outputs = layer.Forward(outputs);
            }
            return outputs.ToArray();
        }

        // Forward pass through the network
        public double[] Forward(double[] inputs)
        {
            Vector<double> outputs = Vector<double>.Build.DenseOfArray(inputs);
            foreach (var layer in Layers)
            {
                outputs = layer.Forward(outputs);
            }
            return outputs.ToArray();
        }

        // Train
        public double Train(double[][] inputs, int[] labels)
        {
            List<double> learningRates = new List<double>();
            double initialLearningRate = LearningRate;

            // Decay learning rate
            learningRates.Add(LearningRate);
            LearningRate = initialLearningRate / (1 + 0.01 * Epoch);

            long intEpochLoss = 0;

            // Shuffle the data for each epoch
            int[] indices = Enumerable.Range(0, inputs.Length).ToArray();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = Rng.Next(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }

            // Create shuffled inputs and labels
            double[][] shuffledInputs = new double[inputs.Length][];
            int[] shuffledLabels = new int[labels.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                shuffledInputs[i] = inputs[indices[i]];
                shuffledLabels[i] = labels[indices[i]];
            }

            // Train in parallel batches
            Parallel.For(0, shuffledInputs.Length / BatchSize, batchIndex =>
            {
                int startIndex = batchIndex * BatchSize;
                int batchSize = Math.Min(BatchSize, shuffledInputs.Length - startIndex);
                double[][] batchInputs = new double[batchSize][];
                int[] batchLabels = new int[batchSize];

                for (int j = 0; j < batchSize; j++)
                {
                    batchInputs[j] = shuffledInputs[startIndex + j];
                    batchLabels[j] = shuffledLabels[startIndex + j];
                }

                Vector<double> output = Vector<double>.Build.DenseOfArray(batchInputs[0]);
                for (int layerIndex = 0; layerIndex < Layers.Count; layerIndex++)
                {
                    output = Layers[layerIndex].Forward(output);
                }

                double batchLoss = CalculateLoss(output.ToArray(), batchLabels[0]);
                Interlocked.Add(ref intEpochLoss, (long)(batchLoss * 1000000)); // Multiply by 1,000,000 to preserve decimal places

                double[] layerErrors = CalculateOutputLayerError(output.ToArray(), batchLabels[0]);

                for (int layerIndex = Layers.Count - 1; layerIndex >= 0; layerIndex--)
                {
                    double[] layerInputs = layerIndex == 0 ? batchInputs[0] : Layers[layerIndex - 1].Outputs.ToArray();
                    layerErrors = Layers[layerIndex].Backward(layerInputs, Layers[layerIndex].Outputs.ToArray(), layerErrors);
                }
            });

            double epochLoss = intEpochLoss / (1000000.0 * inputs.Length); // Divide by 1,000,000 to get back the original scale
            Epoch++;
            Loss.Add(epochLoss);
            return epochLoss;
        }

        // Calculate the loss using cross-entropy
        private double CalculateLoss(double[] outputs, int label)
        {
            return Mathf.CrossEntropyLoss(outputs, label);
        }

        // Calculate the error for the output layer
        private double[] CalculateOutputLayerError(double[] output, int label)
        {
            double[] errors = new double[output.Length];
            for (int i = 0; i < output.Length; i++)
            {
                errors[i] = output[i] - (i == label ? 1.0 : 0.0);
            }
            return errors;
        }
    }
}