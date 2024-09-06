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
	NetworkLearnData[] batchLearnData;
	HyperParameters hyperParameters;

	// Create the neural network
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
		SetActivationFunction(Activation.Activation.GetActivationFromType(hyperParameters.activationType), Activation.Activation.GetActivationFromType(hyperParameters.outputActivationType));
	}

	// Run the inputs through the network to predict which class they belong to.
	// Also returns the activations from the output layer.
	public (int predictedClass, double[] outputs) Classify(double[] inputs)
	{
		var outputs = CalculateOutputs(inputs);
		int predictedClass = MaxValueIndex(outputs);
		return (predictedClass, outputs);
	}

	// Run the inputs through the network to calculate the outputs
	public double[] CalculateOutputs(double[] inputs)
	{
		foreach (Layer layer in layers)
		{
			inputs = layer.CalculateOutputs(inputs);
		}
		return inputs;
	}


	public void Learn(DataPoint[] trainingData, double learnRate, double regularization = 0, double momentum = 0)
	{
		Console.WriteLine($"Starting training with {trainingData.Length} data points");

		Stopwatch stopwatch = new Stopwatch();
		stopwatch.Start();

		double previousLoss = double.MaxValue;
		double currentLoss = 0;

		for (int epoch = 0; epoch < hyperParameters.epochs; epoch++)
		{
			Stopwatch testInTrainingStopwatch = new Stopwatch();
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
				layers[i].ApplyGradients(learnRate, regularization, momentum);
			}

			for (int i = 0; i < trainingData.Length; i++)
			{
				var outputs = CalculateOutputs(trainingData[i].inputs);
				currentLoss += cost.CalculateCost(outputs, trainingData[i].expectedOutputs);
			}
			currentLoss /= trainingData.Length;

			epochStopwatch.Stop();
			testInTrainingStopwatch.Start();

			double trainingAccuracy = Test(trainingData, false);

			testInTrainingStopwatch.Stop();
			Console.WriteLine($"Epoch {epoch + 1}/{hyperParameters.epochs}:");
			Console.WriteLine($"  Loss: {currentLoss:F6}");
			Console.WriteLine($"  Training accuracy: {trainingAccuracy:P2}");
			Console.WriteLine($"  Time: {epochStopwatch.ElapsedMilliseconds}ms");
			Console.WriteLine($"  Loss change: {previousLoss - currentLoss:F6}");
			Console.WriteLine($"  Test in training time: {testInTrainingStopwatch.ElapsedMilliseconds}ms");

			previousLoss = currentLoss;
		}

		stopwatch.Stop();
		Console.WriteLine("Training completed");
		Console.WriteLine($"Total training time: {stopwatch.ElapsedMilliseconds}ms");
		Console.WriteLine($"Final loss: {currentLoss:F6}");
	}

	public double Test(DataPoint[] testData, bool verbose = true)
	{
		int correct = 0;
		foreach (DataPoint data in testData)
		{
			var (predictedClass, outputs) = Classify(data.inputs);
			if (predictedClass == data.expectedOutputs.Max())
			{
				correct++;
			}
		}
		double accuracy = (double)correct / testData.Length;
		if (verbose)
		{
			Console.WriteLine($"Test results: {correct} correct out of {testData.Length}");
		}
		return accuracy;
	}


	void UpdateGradients(DataPoint data, NetworkLearnData learnData)
	{
		// Feed data through the network to calculate outputs.
		// Save all inputs/weightedinputs/activations along the way to use for backpropagation.
		double[] inputsToNextLayer = data.inputs;

		for (int i = 0; i < layers.Length; i++)
		{
			inputsToNextLayer = layers[i].CalculateOutputs(inputsToNextLayer, learnData.layerData[i]);
		}

		// -- Backpropagation --
		int outputLayerIndex = layers.Length - 1;
		Layer outputLayer = layers[outputLayerIndex];
		LayerLearnData outputLearnData = learnData.layerData[outputLayerIndex];

		// Update output layer gradients
		outputLayer.CalculateOutputLayerNodeValues(outputLearnData, data.expectedOutputs, cost);
		outputLayer.UpdateGradients(outputLearnData);

		// Update all hidden layer gradients
		for (int i = outputLayerIndex - 1; i >= 0; i--)
		{
			LayerLearnData layerLearnData = learnData.layerData[i];
			Layer hiddenLayer = layers[i];

			hiddenLayer.CalculateHiddenLayerNodeValues(layerLearnData, layers[i + 1], learnData.layerData[i + 1].nodeValues);
			hiddenLayer.UpdateGradients(layerLearnData);
		}

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
		double maxValue = double.MinValue;
		int index = 0;
		for (int i = 0; i < values.Length; i++)
		{
			if (values[i] > maxValue)
			{
				maxValue = values[i];
				index = i;
			}
		}

		return index;
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