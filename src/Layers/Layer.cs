using NeuralNetwork.src.Utils;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.src.Layers
{
    public class Layer
    {
        public Activation Activation { get; set; }
        public int InputSize { get; set; }
        public int OutputSize { get; private set; } 
        public double[] Outputs { get; set; }
        public double[] Weights;
        public double[] Biases;
        public double LearningRate { get; set; }
        public double RegularizationStrength { get; set; }
        private IOptimizer optimizer;

        private Matrix<double> WeightMatrix;
        private Vector<double> BiasVector;

        public Layer(Activation activation, int inputSize, int outputSize, double[] weights, double[] biases, double learningRate, System.Random rng, double regularizationStrength, IOptimizer optimizer)
        {
            Activation = activation;
            InputSize = inputSize;
            OutputSize = outputSize;
            Weights = weights ?? throw new ArgumentNullException(nameof(weights)); 
            Biases = biases ?? throw new ArgumentNullException(nameof(biases));
            LearningRate = learningRate;
            Outputs = new double[outputSize];
            RegularizationStrength = regularizationStrength;
            this.optimizer = optimizer;

            // Initialize weights and biases with random values
            InitializeRandomWeights(rng);
            WeightMatrix = Matrix<double>.Build.Dense(OutputSize, InputSize, (i, j) => Weights[i * InputSize + j]);
            BiasVector = Vector<double>.Build.Dense(Biases);
        }

        public Vector<double> Forward(Vector<double> input)
        {
            Vector<double> output = WeightMatrix * input + BiasVector;
            output = output.Map(ApplyActivation);

            if (Activation == Activation.Softmax)
            {
                output = Vector<double>.Build.DenseOfArray(Mathf.Softmax(output.ToArray()));
            }

            Outputs = output.ToArray();
            return output;
        }

        private double ApplyActivation(double x)
        {
            switch (Activation)
            {
                case Activation.ReLU:
                    return Mathf.ReLU(x);
                case Activation.Sigmoid:
                    return Mathf.Sigmoid(x);
                case Activation.Tanh:
                    return Mathf.Tanh(x);
                case Activation.Softmax:
                    return Math.Exp(x);
                default:
                    throw new ArgumentException("Unsupported activation function.");
            }
        }

        private double ApplyActivationDerivative(double x)
        {
            switch (Activation)
            {
                case Activation.ReLU:
                    return x > 0 ? 1 : 0;
                case Activation.Sigmoid:
                    return x * (1 - x);
                case Activation.Softmax:
                    return x * (1 - x); 
                case Activation.Tanh:
                    return 1 - Math.Pow(x, 2);
                default:
                    throw new ArgumentException("Unsupported activation function.");
            }
        }

        public double[] Backward(double[] inputs, double[] outputs, double[] errors)
        {
            // Ensure all vectors have the same dimensionality
            if (inputs.Length != InputSize || outputs.Length != OutputSize || errors.Length != OutputSize)
            {
                throw new ArgumentException("All vectors must have the same dimensionality.");
            }

            Vector<double> inputVector = Vector<double>.Build.DenseOfArray(inputs);
            Vector<double> outputVector = Vector<double>.Build.DenseOfArray(outputs);
            Vector<double> errorVector = Vector<double>.Build.DenseOfArray(errors);

            Vector<double> delta = errorVector.PointwiseMultiply(outputVector.Map(ApplyActivationDerivative));
            Matrix<double> weightGradient = delta.OuterProduct(inputVector);
            Vector<double> biasGradient = delta;

            optimizer.UpdateWeights(ref Weights, weightGradient.ToColumnMajorArray());
            optimizer.UpdateBiases(ref Biases, biasGradient.ToArray());

            return weightGradient.ToColumnMajorArray();
        }

        private void InitializeRandomWeights(System.Random rng)
        {
		    for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = RandomInNormalDistribution(rng, 0, 1) / Math.Sqrt(InputSize);
            }

            static double RandomInNormalDistribution(System.Random rng, double mean, double standardDeviation)
            {
                double x1 = 1 - rng.NextDouble();
                double x2 = 1 - rng.NextDouble();

                double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
                return y1 * standardDeviation + mean;
            }
        }

    }
}