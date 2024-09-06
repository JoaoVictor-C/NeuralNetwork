using NeuralNetwork.Activation;
using NeuralNetwork.Cost;
using DataHandling;
using Utils;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Numerics;

namespace NeuralNetwork;

public class NeuralNetwork
{
    public readonly Layer[] layers;
    public readonly int[] layerSizes;

    public ICost cost;
    System.Random rng;
    NetworkLearnData[] batchLearnData;
    HyperParameters hyperParameters;

    public NeuralNetwork(int[] layerSizes)
    {
        hyperParameters = HyperParameters.LoadFromJson("src/NeuralNetwork/parameters.json");

        batchLearnData = new NetworkLearnData[0];
        this.layerSizes = layerSizes;
        rng = new System.Random();

        layers = new Layer[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], rng);
        }
        cost = Cost.Cost.GetCostFromType(hyperParameters.costType);
        SetActivationFunction(
            Activation.Activation.GetActivationFromType(hyperParameters.activationType),
            Activation.Activation.GetActivationFromType(hyperParameters.outputActivationType)
        );
    }

    public (int predictedClass, double[] outputs) Classify(double[] inputs)
    {
        var outputs = CalculateOutputs(inputs);
        int predictedClass = MaxValueIndex(outputs);
        return (predictedClass, outputs);
    }

    public double[] CalculateOutputs(double[] inputs)
    {
        foreach (Layer layer in layers)
        {
            inputs = layer.CalculateOutputs(inputs);
        }

        return inputs;
    }

    public void Learn(DataPoint[] trainingData, double initialLearnRate, double regularization = 0, double momentum = 0)
    {
        Console.WriteLine($"Starting training with {trainingData.Length} data points");

        Stopwatch stopwatch = new Stopwatch();
        Stopwatch testStopwatch = new Stopwatch();
        stopwatch.Start();
        testStopwatch.Start();

        double previousLoss = 1;
        double currentLoss = 0;

        ParallelOptions parallelOptions = new ParallelOptions();
        parallelOptions.MaxDegreeOfParallelism = Environment.ProcessorCount;

        DataPoint[] shuffledTrainData = trainingData.OrderBy(x => rng.Next()).ToArray();
        int validationSize = (int)(shuffledTrainData.Length * 0.2);
        DataPoint[] validationData = shuffledTrainData.Take(validationSize).ToArray();
        shuffledTrainData = shuffledTrainData.Skip(validationSize).ToArray();
        DataPoint[] testData = shuffledTrainData.Take(10000).ToArray();

        double learnRate = initialLearnRate;

        double bestValidationAccuracy = 0;
        int patienceCounter = 0;
        int maxPatience = 5;

        for (int epoch = 0; epoch < hyperParameters.epochs; epoch++)
        {
            learnRate = initialLearnRate / (1 + hyperParameters.learnRateDecay * epoch);
            shuffledTrainData = shuffledTrainData.OrderBy(x => rng.Next()).ToArray();

            Parallel.For(0, shuffledTrainData.Length / hyperParameters.minibatchSize, parallelOptions, batchIndex =>
            {
                int startIndex = batchIndex * hyperParameters.minibatchSize;
                int batchSize = Math.Min(hyperParameters.minibatchSize, shuffledTrainData.Length - startIndex);
                var batch = shuffledTrainData.Skip(startIndex).Take(batchSize).ToArray();

                ProcessMiniBatch(batch, learnRate, regularization, momentum);
            });

            currentLoss = CalculateLoss(shuffledTrainData);

            testStopwatch.Restart();
            double trainingAccuracy = Test(testData, false);
            testStopwatch.Stop();

            double validationAccuracy = Test(validationData, false);
            if (validationAccuracy > bestValidationAccuracy)
            {
                bestValidationAccuracy = validationAccuracy;
                patienceCounter = 0;
            }
            else
            {
                patienceCounter++;
                if (patienceCounter >= maxPatience)
                {
                    Console.WriteLine($"Early stopping at epoch {epoch + 1}");
                    break;
                }
            }

            Console.WriteLine($"Epoch {epoch + 1}/{hyperParameters.epochs}:");
            Console.WriteLine($"  Loss: {currentLoss:F3}");
            Console.WriteLine($"  Training accuracy: {trainingAccuracy:P2}");
            Console.WriteLine($"  Validation accuracy: {validationAccuracy:P2}");
            Console.WriteLine($"  Time: {stopwatch.ElapsedMilliseconds / 1000}s");
            Console.WriteLine($"  Loss change: {(previousLoss - currentLoss) * -1:F3}");
            Console.WriteLine($"  Test in training time: {testStopwatch.ElapsedMilliseconds / 1000}s");

            previousLoss = currentLoss;
        }

        stopwatch.Stop();
        Console.WriteLine("Training completed");
        Console.WriteLine($"Total training time: {stopwatch.ElapsedMilliseconds / 1000}s");
        Console.WriteLine($"Final loss: {currentLoss:F6}");
    }

    private void ProcessMiniBatch(DataPoint[] batch, double learnRate, double regularization, double momentum)
    {
        var networkGradients = new NetworkGradients(layers);

        foreach (var dataPoint in batch)
        {
            ForwardPropagate(dataPoint.inputs, out var layerOutputs);
            BackPropagate(dataPoint, layerOutputs, networkGradients);
        }

        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].ApplyGradients(networkGradients.layerGradients[i], learnRate / batch.Length, regularization, momentum);
        }
    }

    private double CalculateLoss(DataPoint[] data)
    {
        ParallelOptions parallelOptions = new ParallelOptions();
        parallelOptions.MaxDegreeOfParallelism = Environment.ProcessorCount;

        double totalLoss = 0;
        Parallel.For(0, data.Length, parallelOptions, i =>
        {
            var outputs = CalculateOutputs(data[i].inputs);
            double loss = cost.CalculateCost(outputs, data[i].expectedOutputs);
            Interlocked.Exchange(ref totalLoss, totalLoss + loss);
        });
        return totalLoss / data.Length;
    }

    public double Test(DataPoint[] testData, bool verbose = true)
    {
        int correct = 0;
        foreach (DataPoint data in testData)
        {
            var (predictedClass, outputs) = Classify(data.inputs);
            if (predictedClass == MaxValueIndex(data.expectedOutputs))
            {
                correct++;
            }
        }
        double accuracy = (double)correct / testData.Length;
        if (verbose)
        {
            Console.WriteLine($"Test results: {correct} correct out of {testData.Length}");
            Console.WriteLine($"Test accuracy: {accuracy:P2}");
        }
        return accuracy;
    }

    private void ForwardPropagate(double[] inputs, out double[][] layerOutputs)
    {
        layerOutputs = new double[layers.Length + 1][];
        layerOutputs[0] = inputs;

        var outputs = layerOutputs;
        Parallel.For(0, layers.Length, i =>
        {
            outputs[i + 1] = layers[i].CalculateOutputs(outputs[i]);
        });
    }

    private void BackPropagate(DataPoint dataPoint, double[][] layerOutputs, NetworkGradients networkGradients)
    {
        var networkLearnData = new NetworkLearnData(layers);

        int outputLayerIndex = layers.Length - 1;
        layers[outputLayerIndex].CalculateOutputLayerNodeValues(networkLearnData.layerData[outputLayerIndex], dataPoint.expectedOutputs, cost);

        Parallel.For(0, outputLayerIndex, i =>
        {
            int layerIndex = outputLayerIndex - i - 1;
            layers[layerIndex].CalculateHiddenLayerNodeValues(networkLearnData.layerData[layerIndex], layers[layerIndex + 1], networkLearnData.layerData[layerIndex + 1].nodeValues);
        });

        Parallel.For(0, layers.Length, i =>
        {
            networkLearnData.layerData[i].inputs = layerOutputs[i];
            layers[i].UpdateGradients(networkLearnData.layerData[i]);
        });
    }

    public void SetCostFunction(ICost costFunction)
    {
        this.cost = costFunction;
    }

    public void SetActivationFunction(IActivation activation)
    {
        SetActivationFunction(activation, activation);
    }

    public void SetActivationFunction(IActivation activation, IActivation outputLayerActivation)
    {
        for (int i = 0; i < layers.Length - 1; i++)
        {
            layers[i].SetActivationFunction(activation);
        }
        layers[layers.Length - 1].SetActivationFunction(outputLayerActivation);
    }

    public int MaxValueIndex(double[] values)
    {
        double maxValue = values.Max();
        int maxIndex = Array.IndexOf(values, maxValue);
        return maxIndex;
    }
}

public class NetworkLearnData
{
    public LayerLearnData[] layerData;

    public NetworkLearnData(Layer[] layers)
    {
        layerData = new LayerLearnData[layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            layerData[i] = new LayerLearnData(layers[i]);
        }
    }
}

public class LayerLearnData
{
    public double[] inputs;
    public double[] weightedInputs;
    public double[] activations;
    public double[] nodeValues;

    public LayerLearnData(Layer layer)
    {
        weightedInputs = new double[layer.numNodesOut];
        activations = new double[layer.numNodesOut];
        nodeValues = new double[layer.numNodesOut];
    }
}

public class NetworkGradients
{
    public LayerGradients[] layerGradients;

    public NetworkGradients(Layer[] layers)
    {
        layerGradients = new LayerGradients[layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            layerGradients[i] = new LayerGradients(layers[i]);
        }
    }
}

public class LayerGradients
{
    public double[] costGradientW;
    public double[] costGradientB;

    public LayerGradients(Layer layer)
    {
        costGradientW = new double[layer.weights.Length];
        costGradientB = new double[layer.biases.Length];
    }
}