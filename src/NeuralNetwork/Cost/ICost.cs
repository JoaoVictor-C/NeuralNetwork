namespace NeuralNetwork.Cost;

public interface ICost
{
	double CostFunction(double[] predictedOutputs, double[] expectedOutputs);
	double CostDerivative(double output, double expectedOutput);
	double CalculateCost(double[] outputs, double[] expectedOutputs);
	Cost.CostType CostFunctionType();
}