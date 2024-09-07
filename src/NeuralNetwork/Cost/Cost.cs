using System;

namespace NeuralNetwork.Cost;

public class Cost
{

	public enum CostType
	{
		MeanSquareError,
		CrossEntropy
	}

	public static ICost GetCostFromType(CostType type)
	{
		switch (type)
		{
			case CostType.MeanSquareError:
				return new MeanSquaredError();
			case CostType.CrossEntropy:
				return new CrossEntropy();
			default:
				Console.WriteLine("Unhandled cost type");
				return new MeanSquaredError();
		}
	}

	public class MeanSquaredError : ICost
	{
		public double CalculateCost(double[] outputs, double[] expectedOutputs)
		{
			return CostFunction(outputs, expectedOutputs);
		}

		public double CostFunction(double[] predictedOutputs, double[] expectedOutputs)
		{
			// cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
			double cost = 0;
			for (int i = 0; i < predictedOutputs.Length; i++)
			{
				double error = predictedOutputs[i] - expectedOutputs[i];
				cost += error * error;
			}
			return 0.5 * cost;
		}

		public double CostDerivative(double predictedOutput, double expectedOutput)
		{
			return predictedOutput - expectedOutput;
		}

		public CostType CostFunctionType()
		{
			return CostType.MeanSquareError;
		}
	}

	public class CrossEntropy : ICost
	{
		private const double epsilon = 1e-15; // Small value to prevent log(0)

		public double CalculateCost(double[] outputs, double[] expectedOutputs)
		{
			return CostFunction(outputs, expectedOutputs);
		}

		public double CostFunction(double[] predictedOutputs, double[] expectedOutputs)
        {
            double cost = 0;
            for (int i = 0; i < predictedOutputs.Length; i++)
            {
                cost += expectedOutputs[i] * Math.Log(predictedOutputs[i]) + (1 - expectedOutputs[i]) * Math.Log(1 - predictedOutputs[i]);
            }
            return -cost;
        }

		public double CostDerivative(double predictedOutput, double expectedOutput)
		{
			return (predictedOutput - expectedOutput) / (predictedOutput * (1 - predictedOutput));
		}

		public CostType CostFunctionType()
		{
			return CostType.CrossEntropy;
		}
	}

}