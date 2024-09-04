namespace NeuralNetwork.src.Utils
{
    public enum OptimizerType
    {
        SGD,
        Momentum,
        Adam,
        RMSprop
    }

    public interface IOptimizer
    {
        void UpdateWeights(ref double[] weights, double[] weightGradients);
        void UpdateBiases(ref double[] biases, double[] biasGradients);
        IOptimizer Clone();
    }

    public class SGD : IOptimizer
    {
        private readonly double learningRate;

        public SGD(double learningRate)
        {
            this.learningRate = learningRate;
        }

        public void UpdateWeights(ref double[] weights, double[] weightGradients)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] -= learningRate * weightGradients[i];
            }
        }

        public void UpdateBiases(ref double[] biases, double[] biasGradients)
        {
            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] -= learningRate * biasGradients[i];
            }
        }

        public IOptimizer Clone() => new SGD(learningRate);
    }

    public class Momentum : IOptimizer
    {
        private readonly double learningRate;
        private readonly double momentum;
        private double[] velocityWeights;
        private double[] velocityBiases;

        public Momentum(double learningRate, double momentum, int weightSize, int biasSize)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            velocityWeights = new double[weightSize];
            velocityBiases = new double[biasSize];
        }

        public void UpdateWeights(ref double[] weights, double[] weightGradients)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                velocityWeights[i] = momentum * velocityWeights[i] - learningRate * weightGradients[i];
                weights[i] += velocityWeights[i];
            }
        }

        public void UpdateBiases(ref double[] biases, double[] biasGradients)
        {
            for (int i = 0; i < biases.Length; i++)
            {
                velocityBiases[i] = momentum * velocityBiases[i] - learningRate * biasGradients[i];
                biases[i] += velocityBiases[i];
            }
        }

        public IOptimizer Clone() => new Momentum(learningRate, momentum, velocityWeights.Length, velocityBiases.Length);
    }

    public class Adam : IOptimizer
    {
        private readonly double learningRate;
        private readonly double beta1;
        private readonly double beta2;
        private readonly double epsilon;
        private double[] mWeights;
        private double[] vWeights;
        private double[] mBiases;
        private double[] vBiases;
        private int t;

        public Adam(double learningRate, double beta1, double beta2, double epsilon, int weightSize, int biasSize)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            mWeights = new double[weightSize];
            vWeights = new double[weightSize];
            mBiases = new double[biasSize];
            vBiases = new double[biasSize];
            t = 0;
        }

        public void UpdateWeights(ref double[] weights, double[] weightGradients)
        {
            t++;
            for (int i = 0; i < weights.Length; i++)
            {
                mWeights[i] = beta1 * mWeights[i] + (1 - beta1) * weightGradients[i];
                vWeights[i] = beta2 * vWeights[i] + (1 - beta2) * weightGradients[i] * weightGradients[i];
                double mHat = mWeights[i] / (1 - Math.Pow(beta1, t));
                double vHat = vWeights[i] / (1 - Math.Pow(beta2, t));
                weights[i] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
            }
        }

        public void UpdateBiases(ref double[] biases, double[] biasGradients)
        {
            for (int i = 0; i < biases.Length; i++)
            {
                mBiases[i] = beta1 * mBiases[i] + (1 - beta1) * biasGradients[i];
                vBiases[i] = beta2 * vBiases[i] + (1 - beta2) * biasGradients[i] * biasGradients[i];
                double mHat = mBiases[i] / (1 - Math.Pow(beta1, t));
                double vHat = vBiases[i] / (1 - Math.Pow(beta2, t));
                biases[i] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
            }
        }

        public IOptimizer Clone() => new Adam(learningRate, beta1, beta2, epsilon, mWeights.Length, mBiases.Length);
    }

    public class RMSprop : IOptimizer
    {
        private readonly double learningRate;
        private readonly double beta;
        private readonly double epsilon;
        private double[] cacheWeights;
        private double[] cacheBiases;

        public RMSprop(double learningRate, double beta, double epsilon, int weightSize, int biasSize)
        {
            this.learningRate = learningRate;
            this.beta = beta;
            this.epsilon = epsilon;
            cacheWeights = new double[weightSize];
            cacheBiases = new double[biasSize];
        }

        public void UpdateWeights(ref double[] weights, double[] weightGradients)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                cacheWeights[i] = beta * cacheWeights[i] + (1 - beta) * weightGradients[i] * weightGradients[i];
                weights[i] -= learningRate * weightGradients[i] / (Math.Sqrt(cacheWeights[i]) + epsilon);
            }
        }

        public void UpdateBiases(ref double[] biases, double[] biasGradients)
        {
            for (int i = 0; i < biases.Length; i++)
            {
                cacheBiases[i] = beta * cacheBiases[i] + (1 - beta) * biasGradients[i] * biasGradients[i];
                biases[i] -= learningRate * biasGradients[i] / (Math.Sqrt(cacheBiases[i]) + epsilon);
            }
        }

        public IOptimizer Clone() => new RMSprop(learningRate, beta, epsilon, cacheWeights.Length, cacheBiases.Length);
    }
}
