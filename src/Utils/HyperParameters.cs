using Newtonsoft.Json;
using NeuralNetwork.Activation;
using NeuralNetwork.Cost;

namespace Utils;

public class HyperParameters
{
    public NeuralNetwork.Activation.Activation.ActivationType activationType;
    public NeuralNetwork.Activation.Activation.ActivationType outputActivationType;
    public NeuralNetwork.Cost.Cost.CostType costType;
    public double initialLearningRate;
    public double learnRateDecay;
    public int minibatchSize;
    public double momentum;
    public double regularization;
    public int[] layerSizes;
    public int epochs;
    public double trainTestSplit;
    public static HyperParameters LoadFromJson(string filePath)
    {
        string json = File.ReadAllText(filePath);
        return JsonConvert.DeserializeObject<HyperParameters>(json);
    }
}