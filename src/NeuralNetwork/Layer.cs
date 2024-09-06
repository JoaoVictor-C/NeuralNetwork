using static System.Math;
using Utils;
using System.Numerics;

namespace NeuralNetwork;

public class Layer
{
    public readonly int numNodesIn;
    public readonly int numNodesOut;

    public readonly double[] weights;
    public readonly double[] biases;

    public readonly double[] costGradientW;
    public readonly double[] costGradientB;

    public readonly double[] weightVelocities;
    public readonly double[] biasVelocities;

    public Activation.IActivation activation;

    private HyperParameters hyperParameters;

    private double[] mW, vW, mB, vB;
    private const double beta1 = 0.9;
    private const double beta2 = 0.999;
    private const double epsilon = 1e-8;
    private int t = 0;

    public Layer(int numNodesIn, int numNodesOut, System.Random rng)
    {
        hyperParameters = HyperParameters.LoadFromJson("src/NeuralNetwork/parameters.json");

        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        activation = Activation.Activation.GetActivationFromType(hyperParameters.activationType);

        weights = new double[numNodesIn * numNodesOut];
        costGradientW = new double[weights.Length];
        biases = new double[numNodesOut];
        costGradientB = new double[biases.Length];

        weightVelocities = new double[weights.Length];
        biasVelocities = new double[biases.Length];

        mW = new double[weights.Length];
        vW = new double[weights.Length];
        mB = new double[biases.Length];
        vB = new double[biases.Length];

        InitializeRandomWeights(rng);
    }

    public double[] CalculateOutputs(double[] inputs, bool training = false, double dropoutRate = 0.5)
    {
        double[] weightedInputs = new double[numNodesOut];
        double[] activations = new double[numNodesOut];
        System.Random rng = new System.Random();

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double weightedInput = biases[nodeOut];
            int vectorSize = Vector<double>.Count;

            for (int nodeIn = 0; nodeIn < numNodesIn - vectorSize + 1; nodeIn += vectorSize)
            {
                var inputVector = new Vector<double>(inputs, nodeIn);
                var weightVector = new Vector<double>(weights, GetFlatWeightIndex(nodeIn, nodeOut));
                weightedInput += Vector.Dot(inputVector, weightVector);
            }

            for (int nodeIn = numNodesIn - (numNodesIn % vectorSize); nodeIn < numNodesIn; nodeIn++)
            {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }

            weightedInputs[nodeOut] = weightedInput;
        }

        for (int outputNode = 0; outputNode < numNodesOut; outputNode++)
        {
            activations[outputNode] = activation.Activate(weightedInputs, outputNode);
        }

        if (training)
        {
            for (int i = 0; i < activations.Length; i++)
            {
                if (rng.NextDouble() < dropoutRate)
                {
                    activations[i] = 0;
                }
            }
        }

        return activations;
    }

    public double[] CalculateOutputs(double[] inputs, LayerLearnData learnData)
    {
        learnData.inputs = inputs;

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double weightedInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }
            learnData.weightedInputs[nodeOut] = weightedInput;
        }

        for (int i = 0; i < learnData.activations.Length; i++)
        {
            learnData.activations[i] = activation.Activate(learnData.weightedInputs, i);
        }

        return learnData.activations;
    }

        public void ApplyGradients(LayerGradients gradients, double learnRate, double regularization, double momentum)
    {
        t++;
        double weightDecay = (1 - regularization * learnRate);

        for (int i = 0; i < weights.Length; i++)
        {
            mW[i] = beta1 * mW[i] + (1 - beta1) * gradients.costGradientW[i];
            vW[i] = beta2 * vW[i] + (1 - beta2) * gradients.costGradientW[i] * gradients.costGradientW[i];

            double mWCorrected = mW[i] / (1 - Math.Pow(beta1, t));
            double vWCorrected = vW[i] / (1 - Math.Pow(beta2, t));

            weights[i] = weightDecay * weights[i] - learnRate * mWCorrected / (Math.Sqrt(vWCorrected) + epsilon);
        }

        for (int i = 0; i < biases.Length; i++)
        {
            mB[i] = beta1 * mB[i] + (1 - beta1) * gradients.costGradientB[i];
            vB[i] = beta2 * vB[i] + (1 - beta2) * gradients.costGradientB[i] * gradients.costGradientB[i];

            double mBCorrected = mB[i] / (1 - Math.Pow(beta1, t));
            double vBCorrected = vB[i] / (1 - Math.Pow(beta2, t));

            biases[i] -= learnRate * mBCorrected / (Math.Sqrt(vBCorrected) + epsilon);
        }

        Array.Clear(gradients.costGradientW, 0, gradients.costGradientW.Length);
        Array.Clear(gradients.costGradientB, 0, gradients.costGradientB.Length);
    }


    public void CalculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, Cost.ICost cost)
    {
        for (int i = 0; i < layerLearnData.nodeValues.Length; i++)
        {
            double costDerivative = cost.CostDerivative(layerLearnData.activations[i], expectedOutputs[i]);
            double activationDerivative = activation.Derivative(layerLearnData.weightedInputs, i);
            layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
        }
    }

    public void CalculateHiddenLayerNodeValues(LayerLearnData layerLearnData, Layer oldLayer, double[] oldNodeValues)
    {
        for (int newNodeIndex = 0; newNodeIndex < numNodesOut; newNodeIndex++)
        {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.Length; oldNodeIndex++)
            {
                double weightedInputDerivative = oldLayer.GetWeight(newNodeIndex, oldNodeIndex);
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= activation.Derivative(layerLearnData.weightedInputs, newNodeIndex);
            layerLearnData.nodeValues[newNodeIndex] = newNodeValue;
        }
    }

    public void UpdateGradients(LayerLearnData layerLearnData)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double nodeValue = layerLearnData.nodeValues[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                int weightIndex = GetFlatWeightIndex(nodeIn, nodeOut);
                double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
                costGradientW[weightIndex] += derivativeCostWrtWeight;
            }
            costGradientB[nodeOut] += nodeValue;
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
            weights[i] = RandomInNormalDistribution(rng, 0, Sqrt(2.0 / (numNodesIn + numNodesOut)));
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