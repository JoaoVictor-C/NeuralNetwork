 using NeuralNetwork.src.Utils;
using NeuralNetwork.src.Models;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            RunNeuralNetwork();
            //GenerateAugmentedData(1000000);
        }

        static void RunNeuralNetwork() {
            try
            {
                // Clean the console
                Console.Clear();

                // Define the neural network parameters
                int[] layerSizes = new int[] { 784, 128, 64, 32, 10 };
                Activation[] activations = new Activation[] { Activation.ReLU, Activation.ReLU, Activation.ReLU, Activation.ReLU, Activation.Softmax };
                double learningRate = 0.001;
                int epochs = 20;
                int batchSize = 64;
                int verbose = 1;
                double regularizationStrength = 1e-8;
                int numModels = 3;
                int numTrainingSamples = 100000;
                int numTestSamples = 100000;
                OptimizerType optimizerType = OptimizerType.Adam;

                // Define the optimizer parameters
                double beta1 = 0.9; 
                double beta2 = 0.99;
                double epsilon = 1e-8;

                InitialMessages(layerSizes, learningRate, epochs, batchSize, verbose, numModels, regularizationStrength);

                // Initialize the neural network with the optimizer
                Ensemble ensemble = new Ensemble(numModels, layerSizes, activations, learningRate, epochs, batchSize, verbose, regularizationStrength, optimizerType);

                TrainEnsemble(ensemble, numTrainingSamples);
                TestEnsemble(ensemble, numTrainingSamples, numTestSamples);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred while loading the data: {ex.Message}");
            }
        }

        static void GenerateAugmentedData(int numSamples) {
            ImageAugmentation.CreateAugmentedDataset("data/augmented", numSamples);
        }

        static void InitialMessages(int[] layerSizes, double learningRate, int epochs, int batchSize, int verbose, int numModels, double regularizationStrength)
        {
            Console.WriteLine("Initializing Neural Network with parameters: ");
            Console.WriteLine($"Layer Sizes: {string.Join(", ", layerSizes)}");
            Console.WriteLine($"Learning Rate: {learningRate}");
            Console.WriteLine($"Epochs: {epochs}");
            Console.WriteLine($"Batch Size: {batchSize}");
            Console.WriteLine($"Verbose: {verbose}");
            Console.WriteLine($"Number of Models: {numModels}");
            Console.WriteLine($"Regularization Strength: {regularizationStrength}");
        }

        static void TrainEnsemble(Ensemble ensemble, int numTrainingSamples)
        {
            var trainingData = MnistReader.ReadAugmentedData().Take(numTrainingSamples).ToList();
            double[][] inputs = trainingData.Select(image => ImageHydratation.FlattenAndNormalize(image.Data)).ToArray();
            int[] labels = trainingData.Select(image => (int)image.Label).ToArray();

            ensemble.Train(inputs, labels);
        }

        static void TestEnsemble(Ensemble ensemble, int numTrainingSamples, int numTestSamples)
        {
            var testData = MnistReader.ReadAugmentedData().Skip(numTrainingSamples).Take(numTestSamples).ToList();
            double[][] testInputs = testData.Select(image => ImageHydratation.FlattenAndNormalize(image.Data)).ToArray();
            int[] testLabels = testData.Select(image => (int)image.Label).ToArray();

            ensemble.Test(testInputs, testLabels);
        }
    }
}