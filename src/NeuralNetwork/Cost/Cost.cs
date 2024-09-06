namespace NeuralNetwork.Cost;

public class Cost
{
    public enum CostType
    {
        MeanSquareError,
        CrossEntropy,
        HuberLoss,
        LogCosh
    }

    public static ICost GetCostFromType(CostType type)
    {
        return type switch
        {
            CostType.MeanSquareError => new MeanSquaredError(),
            CostType.CrossEntropy => new CrossEntropy(),
            CostType.HuberLoss => new HuberLoss(),
            CostType.LogCosh => new LogCosh(),
            _ => throw new ArgumentException("Unhandled cost type", nameof(type))
        };
    }

    public class MeanSquaredError : ICost
    {
        public double CalculateCost(double[] outputs, double[] expectedOutputs)
        {
            return CostFunction(outputs, expectedOutputs);
        }

        public double CostFunction(double[] predictedOutputs, double[] expectedOutputs)
        {
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
            return predictedOutput - expectedOutput;
        }

        public CostType CostFunctionType()
        {
            return CostType.CrossEntropy;
        }
    }

    public class HuberLoss : ICost
    {
        private const double Delta = 1.0;

        public double CalculateCost(double[] outputs, double[] expectedOutputs)
        {
            return CostFunction(outputs, expectedOutputs);
        }

        public double CostFunction(double[] predictedOutputs, double[] expectedOutputs)
        {
            return Enumerable.Range(0, predictedOutputs.Length)
                .Sum(i => HuberLossFunction(predictedOutputs[i], expectedOutputs[i]));
        }

        private double HuberLossFunction(double predicted, double expected)
        {
            double error = Math.Abs(predicted - expected);
            return error <= Delta ? 0.5 * error * error : Delta * (error - 0.5 * Delta);
        }

        public double CostDerivative(double predictedOutput, double expectedOutput)
        {
            double error = predictedOutput - expectedOutput;
            return Math.Abs(error) <= Delta ? error : Delta * Math.Sign(error);
        }

        public CostType CostFunctionType() => CostType.HuberLoss;
    }

    public class LogCosh : ICost
    {
        public double CalculateCost(double[] outputs, double[] expectedOutputs)
        {
            return CostFunction(outputs, expectedOutputs);
        }

        public double CostFunction(double[] predictedOutputs, double[] expectedOutputs)
        {
            return Enumerable.Range(0, predictedOutputs.Length)
                .Sum(i => Math.Log(Math.Cosh(predictedOutputs[i] - expectedOutputs[i])));
        }

        public double CostDerivative(double predictedOutput, double expectedOutput)
        {
            return Math.Tanh(predictedOutput - expectedOutput);
        }

        public CostType CostFunctionType() => CostType.LogCosh;
    }
}