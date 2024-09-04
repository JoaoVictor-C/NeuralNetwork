using System.Collections.Concurrent;
using System.Diagnostics;
using NeuralNetwork.src.Utils;
using NeuralNetwork.src.Models;

namespace NeuralNetwork.src.Models
{
    public class Ensemble
    {
        private List<NeuralNet> models;
        private PerformanceCounter? cpuCounter;
        private PerformanceCounter? ramCounter;

        public Ensemble(int numModels, int[] layerSizes, Activation[] activations, double learningRate, int epochs, int batchSize, int verbose, double regularizationStrength, OptimizerType optimizerType)
        {
            models = new List<NeuralNet>();
            for (int i = 0; i < numModels; i++)
            {
                models.Add(new NeuralNet(layerSizes, activations, learningRate, epochs, batchSize, verbose, new Random(), regularizationStrength, optimizerType));
            }

            // Verify if the system is windows
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows))
            {
                cpuCounter = new PerformanceCounter("Processor", "% Processor Time", "_Total");
                ramCounter = new PerformanceCounter("Memory", "Available MBytes");
            }
        }

        public void Train(double[][] inputs, int[] labels)
        {
            var watch = new Stopwatch();
            watch.Start();

            Console.WriteLine($"Training ensemble of {models.Count} models...");
            int totalEpochs = models[0].Epochs * models.Count;
            int[] completedEpochs = new int[models.Count];
            object[] modelLocks = Enumerable.Range(0, models.Count).Select(_ => new object()).ToArray();
            object progressLock = new object();

            Parallel.For(0, models.Count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, (index) =>
            {
                for (int epoch = 0; epoch < models[index].Epochs; epoch++)
                {
                    var epochWatch = Stopwatch.StartNew();
                    double epochLoss = models[index].Train(inputs, labels);
                    double learningRate = models[index].LearningRate;
                    epochWatch.Stop();

                    // Clean the lines that are used to display the model progress
                    Console.SetCursorPosition(0, index + 10);
                    Console.Write(new string(' ', Console.WindowWidth));

                    lock (modelLocks[index])
                    {
                        Console.SetCursorPosition(0, index + 10);
                        Console.Write($"\rModel {index + 1}: Epoch {epoch + 1}/{models[index].Epochs} | " +
                                        $"Loss: {epochLoss:F4} | Time: {epochWatch.Elapsed.TotalSeconds:F2}s | " +
                                        $"Learning Rate: {learningRate:F6}");
                    }
                    lock (progressLock)
                    {
                        completedEpochs[index]++;
                        double progress = (double)completedEpochs.Sum() / totalEpochs;
                        double elapsedSeconds = watch.Elapsed.TotalSeconds;
                        double estimatedTotalSeconds = elapsedSeconds / progress;
                        double remainingSeconds = estimatedTotalSeconds - elapsedSeconds;

                        // Clear the line
                        Console.SetCursorPosition(0, models.Count + 11);
                        Console.Write(new string(' ', Console.WindowWidth));
                        Console.SetCursorPosition(0, models.Count + 11);
                        Console.Write($"\rProgress: {progress:P2} | Estimated time remaining: {remainingSeconds:F2}s");

                        // Draw progress bar
                        DrawProgressBar(progress);

                        // Display hardware status
                        DisplayHardwareStatus();
                    }
                }
            });

            Console.SetCursorPosition(0, models.Count + 12);
            Console.WriteLine($"\nEnsemble training completed. Total training time: {watch.Elapsed.TotalSeconds:F2} seconds");
        }

        public int Classify(double[] input)
        {
            if (models.Count == 1)
            {
                double[] prediction = models[0].Classify(input);
                return Array.IndexOf(prediction, prediction.Max());
            }

            var predictions = new ConcurrentBag<double[]>();

            Parallel.ForEach(models, (model) =>
            {
                double[] prediction = model.Classify(input);
                predictions.Add(prediction);
            });

            // Average the probabilities from all models
            double[] averagePrediction = new double[predictions.First().Length];
            foreach (var prediction in predictions)
            {
                for (int i = 0; i < averagePrediction.Length; i++)
                {
                    averagePrediction[i] += prediction[i];
                }
            }
            for (int i = 0; i < averagePrediction.Length; i++)
            {
                averagePrediction[i] /= models.Count;
            }

            // Return the class with the highest average probability
            return Array.IndexOf(averagePrediction, averagePrediction.Max());
        }

        public void Test(double[][] testInputs, int[] testLabels)
        {
            int correct = 0;
            int total = testInputs.Length;
            int verboseFactor = total / 100;
            int count = 0;
            Console.WriteLine("\nTesting the model...");
            var watch = new Stopwatch();
            watch.Start();

            object lockObj = new object();

            Parallel.For(0, total, i =>
            {
                int predictedClass = Classify(testInputs[i]);
                int actualClass = testLabels[i];

                if (predictedClass == actualClass)
                {
                    correct++;
                }
                if ((i + 1) % verboseFactor == 0 || i == total - 1)
                {
                    count++;
                    int CountHydratated = count * verboseFactor;

                    double progress = (double)CountHydratated / total;
                    double elapsedSeconds = watch.Elapsed.TotalSeconds;
                    double estimatedTotalSeconds = elapsedSeconds / progress;
                    double remainingSeconds = estimatedTotalSeconds - elapsedSeconds;

                    updateTestLog(progress, remainingSeconds, total, CountHydratated);
                }
            });

            double accuracy = (double)correct / total;
            Console.WriteLine($"\nTest Accuracy: {accuracy:P2} ({correct}/{total})");
        }
        
        private void updateTestLog(double progress, double remainingSeconds, int total, int CountHydratated)
        {
            Console.Write($"\rProgress: {CountHydratated}/{total} ({progress:P2}) | Estimated time remaining: {remainingSeconds:F2}s");

            DrawProgressBar(progress);
            DisplayHardwareStatusString();
        }

        private void DrawProgressBar(double progress)
        {
            int barWidth = 50;
            int filledWidth = (int)(progress * barWidth);
            string progressBar = $"[{new string('■', filledWidth)}{new string('▪', Math.Min(barWidth - filledWidth, 1))}{new string('□', Math.Max(barWidth - filledWidth - 1, 0))}]";
            Console.SetCursorPosition(0, Console.CursorTop + 1);
            Console.Write(progressBar);
            Console.SetCursorPosition(0, Console.CursorTop - 1);
        }

        private void DisplayHardwareStatus()
        {
            if (cpuCounter == null || ramCounter == null)
            {
                return;
            }

            float cpuUsage = cpuCounter.NextValue();
            float availableMemory = ramCounter.NextValue();
            Console.SetCursorPosition(0, models.Count + 10);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, models.Count + 10);
            Console.Write($"\rCPU Usage: {cpuUsage:F2}% | Available Memory: {availableMemory:F2} MB");
        }

        private string DisplayHardwareStatusString()
        {
            if (cpuCounter == null || ramCounter == null)
            {
                return "";
            }

            float cpuUsage = cpuCounter.NextValue();
            float availableMemory = ramCounter.NextValue();
            return $"\rCPU Usage: {cpuUsage:F2}% | Available Memory: {availableMemory:F2} MB";
        }
    }
}