namespace NeuralNetwork.src.Utils
{
    public static class Mathf
    {
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double SigmoidDerivative(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid * (1 - sigmoid);
        }

        public static double ReLU(double x)
        {
            return x > 0 ? x : 0;
        }

        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        public static double TanhDerivative(double x)
        {
            double tanh = Tanh(x);
            return 1 - Math.Pow(tanh, 2);
        }

        public static double FastExp(double x)
        {
            x = 1.0 + x / 256.0;
            x *= x; x *= x; x *= x; x *= x;
            x *= x; x *= x; x *= x; x *= x;
            return x;
        }

        public static double[] Softmax(double[] x)
        {
            double max = x.Max();
            double[] exp = x.Select(xi => FastExp(xi - max)).ToArray();
            double sum = exp.Sum();
            return exp.Select(xi => xi / sum).ToArray();
        }

        public static double SoftmaxDerivative(double x)
        {
            return Math.Exp(x) / Math.Exp(x);
        }

        public static double CrossEntropyLoss(double[] predicted, int actualClass)
        {
            double[] clippedPredicted = predicted.Select(p => Math.Max(1e-15, Math.Min(1 - 1e-15, p))).ToArray();
            return -Math.Log(clippedPredicted[actualClass]);
        }
    }
}