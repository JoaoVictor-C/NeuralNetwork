using static System.Math;
using Utils;

namespace NeuralNetwork;

public class Layer
{
	public readonly int numNodesIn;
	public readonly int numNodesOut;

	public readonly double[] weights;
	public readonly double[] biases;

	// Cost gradient with respect to weights and with respect to biases
	public readonly double[] costGradientW;
	public readonly double[] costGradientB;

	// Used for adding momentum to gradient descent
	public readonly double[] weightVelocities;
	public readonly double[] biasVelocities;

	public Activation.IActivation activation;

    private HyperParameters hyperParameters;

	// Create the layer
	public Layer(int numNodesIn, int numNodesOut, HyperParameters hyperParameters)
	{
		this.numNodesIn = numNodesIn;
		this.numNodesOut = numNodesOut;
		this.hyperParameters = hyperParameters;

		weights = new double[numNodesIn * numNodesOut];
		biases = new double[numNodesOut];

		costGradientW = new double[numNodesIn * numNodesOut];
		costGradientB = new double[numNodesOut];

		weightVelocities = new double[numNodesIn * numNodesOut];
		biasVelocities = new double[numNodesOut];

		activation = Activation.ActivationFactory.CreateActivation(hyperParameters.activationType);
		cost = Cost.CostFactory.CreateCost(hyperParameters.costType);

		// Update cost gradient with respect to biases (lock for multithreading)
		lock (costGradientB)
		{
			for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
			{
				// Evaluate partial derivative: cost / bias
				double derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut];
				costGradientB[nodeOut] += derivativeCostWrtBias;
			}
		}
	}

	public double GetWeight(int nodeIn, int nodeOut)
	{
		int flatIndex = nodeOut * numNodesIn + nodeIn;
		return weights[flatIndex];
	}

	public int GetFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex)
	{
		return outputNeuronIndex * numNodesIn + inputNeuronIndex;
	}

	public void SetActivationFunction(Activation.IActivation activation)
	{
		this.activation = activation;
	}

	public void InitializeRandomWeights(System.Random rng)
	{
		for (int i = 0; i < weights.Length; i++)
		{
			weights[i] = RandomInNormalDistribution(rng, 0, 1) / Sqrt(numNodesIn);
		}

		double RandomInNormalDistribution(System.Random rng, double mean, double standardDeviation)
		{
			double x1 = 1 - rng.NextDouble();
			double x2 = 1 - rng.NextDouble();

			double y1 = Sqrt(-2.0 * Log(x1)) * Cos(2.0 * PI * x2);
			return y1 * standardDeviation + mean;
		}
	}
}