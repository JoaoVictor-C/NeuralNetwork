using NeuralNetwork;
using DataHandling;
using static DataHandling.ImageLoader;
//using ImageDisplay;
using Utils;

namespace Program;

public class Program
{
	public static void Main(string[] args)
	{
		Console.Clear();
		Console.WriteLine("Starting neural network training and testing...");

		HyperParameters hyperParameters = HyperParameters.LoadFromJson("src/NeuralNetwork/parameters.json");
		Console.WriteLine($"Loaded hyperparameters: Epochs={hyperParameters.epochs}, Learning Rate={hyperParameters.initialLearningRate}, Regularization={hyperParameters.regularization}, Momentum={hyperParameters.momentum}");

		string ImagePath = "data/train-images.idx3-ubyte";
		string LabelPath = "data/train-labels.idx1-ubyte";
		//string ImagePath2 = "data/t10k-images.idx3-ubyte";
		//string LabelPath2 = "data/t10k-labels.idx1-ubyte";

		ImageLoader loader = new ImageLoader(new DataFile[] { 
			new DataFile() { 
				imageFile = new TextAsset() { bytes = FileHelper.ReadAllBytes(ImagePath) }, 
				labelFile = new TextAsset() { bytes = FileHelper.ReadAllBytes(LabelPath) } 
			} 
		}, new string[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" }, 28, true);
	
		// We will load all the data from the image loader
		DataPoint[] data = loader.GetAllData();
		// We will separate the data based on the trainTestSplit value
		DataPoint[] trainData = data.Take((int)(data.Length * hyperParameters.trainTestSplit)).ToArray();
		DataPoint[] testData = data.Skip((int)(data.Length * hyperParameters.trainTestSplit)).ToArray();

		// We will train our neural network with the data we loaded
		Console.WriteLine($"Loaded {data.Length} data points");
		Console.WriteLine($"Training set size: {trainData.Length}, Test set size: {testData.Length}");
		Console.WriteLine("Initializing neural network...");
		NeuralNetwork.NeuralNetwork network = new NeuralNetwork.NeuralNetwork(hyperParameters.layerSizes);

		network.Learn(trainData, hyperParameters.initialLearningRate, hyperParameters.regularization, hyperParameters.momentum);

		network.Test(testData);

	}
}
