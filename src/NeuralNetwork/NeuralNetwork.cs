using NeuralNetwork.Activation;
using NeuralNetwork.Cost;
using DataHandling;
using Utils;
using System.Diagnostics;

namespace NeuralNetwork;

public class NeuralNetwork
{
	public readonly Layer[] layers;
	public readonly int[] layerSizes;

	public ICost cost;
	System.Random rng;
	HyperParameters hyperParameters;
	NetworkLearnData[] batchLearnData;

	// Create the neural network
	public NeuralNetwork(int[] layerSizes)
	{
		hyperParameters = HyperParameters.LoadFromJson("src/NeuralNetwork/parameters.json");

		this.layerSizes = layerSizes;
		layers = new Layer[layerSizes.Length - 1];

		rng = new System.Random(hyperParameters.seed);

		for (int i = 0; i < layers.Length; i++)
		{
			layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], rng);
		}

		// Set the output layer activation to the specified output activation type
		layers[layers.Length - 1].SetActivationFunction(Activation.Activation.GetActivationFromType(hyperParameters.outputActivationType));

		cost = Cost.Cost.GetCostFromType(hyperParameters.costType);

		Console.WriteLine($"Neural network created with layer sizes: {string.Join(", ", layerSizes)}");
	}

	public void Learn(DataPoint[] trainingData, double learningRate, double regularization, double momentum)
	{
		Console.WriteLine($"Starting training with {trainingData.Length} data points");
		Console.WriteLine($"Learning rate: {learningRate}, Regularization: {regularization}, Momentum: {momentum}");

		Stopwatch stopwatch = new Stopwatch();
		stopwatch.Start();

		double previousLoss = double.MaxValue;
		double currentLoss = 0;

		DataPoint[] testData = trainingData.Take((int)(trainingData.Length * hyperParameters.trainTestSplit)).ToArray();

		for (int epoch = 0; epoch < hyperParameters.epochs; epoch++)
		{
			Stopwatch epochStopwatch = new Stopwatch();
			epochStopwatch.Start();

			if (batchLearnData == null || batchLearnData.Length != trainingData.Length)
			{
				batchLearnData = new NetworkLearnData[trainingData.Length];
				for (int i = 0; i < batchLearnData.Length; i++)
				{
					batchLearnData[i] = new NetworkLearnData(layers);
				}
			}

			System.Threading.Tasks.Parallel.For(0, trainingData.Length, (i) =>
			{
				UpdateGradients(trainingData[i], batchLearnData[i]);
			});

			// Apply gradients and calculate loss
			currentLoss = 0;
			for (int i = 0; i < layers.Length; i++)
			{
				layers[i].ApplyGradients(learningRate, regularization, momentum);
			}

			for (int i = 0; i < trainingData.Length; i++)
			{
				var outputs = CalculateOutputs(trainingData[i].inputs);
				currentLoss += cost.CalculateCost(outputs, trainingData[i].expectedOutputs);
			}
			currentLoss /= trainingData.Length;

			epochStopwatch.Stop();

			double trainingAccuracy = Test(testData);
			Console.WriteLine($"Epoch {epoch + 1}/{hyperParameters.epochs}:");
			Console.WriteLine($"  Loss: {currentLoss:F6}");
			Console.WriteLine($"  Training accuracy: {trainingAccuracy:P2}");
			Console.WriteLine($"  Time: {epochStopwatch.ElapsedMilliseconds / 1000.0}s");
			Console.WriteLine($"  Loss change: {previousLoss - currentLoss:F6}");

			previousLoss = currentLoss;
		}

		stopwatch.Stop();
		Console.WriteLine("Training completed");
		Console.WriteLine($"Total training time: {stopwatch.ElapsedMilliseconds / 1000.0}s");
		Console.WriteLine($"Final loss: {currentLoss:F6}");
	}

	public double Test(DataPoint[] testData)
	{
		int correctPredictions = 0;

		Parallel.For(0, testData.Length, (i) =>
		{
			double[] outputs = CalculateOutputs(testData[i].inputs);
			int predictedClass = MaxValueIndex(outputs);
			int actualClass = MaxValueIndex(testData[i].expectedOutputs);

			if (predictedClass == actualClass)
			{
				Interlocked.Increment(ref correctPredictions);
			}
		});

		double accuracy = (double)correctPredictions / testData.Length;
		return accuracy;
	}

	private void UpdateGradients(DataPoint dataPoint, NetworkLearnData learnData)
	{
		// Forward pass
		double[] inputs = dataPoint.inputs;
		for (int i = 0; i < layers.Length; i++)
		{
			inputs = layers[i].CalculateOutputs(inputs, learnData.layerData[i]);
		}

		// Backward pass
		int outputLayerIndex = layers.Length - 1;
		layers[outputLayerIndex].CalculateOutputLayerNodeValues(learnData.layerData[outputLayerIndex], dataPoint.expectedOutputs, cost);
		for (int i = outputLayerIndex - 1; i >= 0; i--)
		{
			layers[i].CalculateHiddenLayerNodeValues(learnData.layerData[i], layers[i + 1], learnData.layerData[i + 1].nodeValues);
		}

		// Update gradients
		for (int i = 0; i < layers.Length; i++)
		{
			layers[i].UpdateGradients(learnData.layerData[i]);
		}
	}

	public double[] CalculateOutputs(double[] inputs)
	{
		for (int i = 0; i < layers.Length; i++)
		{
			inputs = layers[i].CalculateOutputs(inputs);
		}
		return inputs;
	}

	public int MaxValueIndex(double[] values)
	{
		double maxValue = values.Max();
		return Array.IndexOf(values, maxValue);
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
	public double[] gradients;
	public LayerLearnData(Layer layer)
	{
		inputs = new double[layer.numNodesIn];
		weightedInputs = new double[layer.numNodesOut];
		activations = new double[layer.numNodesOut];
		nodeValues = new double[layer.numNodesOut];
		gradients = new double[layer.numNodesOut];
	}
}