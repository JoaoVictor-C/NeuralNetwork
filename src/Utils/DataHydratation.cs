using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork.src.Utils
{
    public class Image
    {
        public byte Label { get; set; }
        public byte[,] Data { get; set; }

        public Image()
        {
            Data = new byte[28, 28]; // Assuming MNIST images are 28x28
        }
    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }



    public static class MnistReader
    {
        private const string TrainImages = "data/train-images.idx3-ubyte";
        private const string TrainLabels = "data/train-labels.idx1-ubyte";
        private const string TestImages = "data/t10k-images.idx3-ubyte";
        private const string TestLabels = "data/t10k-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData()
        {
            return Read(TrainImages, TrainLabels);
        }

        public static IEnumerable<Image> ReadTestData()
        {
            return Read(TestImages, TestLabels);
        }

        public static IEnumerable<Image> ReadOriginalData()
        {
            var trainingData = ReadTrainingData().ToList();
            var testData = ReadTestData().ToList();
            trainingData.AddRange(testData);
            return trainingData;
        }

        public static IEnumerable<Image> ReadAugmentedData()
        {
            string imagesPath = Path.Combine("data", "augmented", "augmented_images.bin");
            string labelsPath = Path.Combine("data", "augmented", "augmented_labels.bin");
            return Read(imagesPath, labelsPath).ToList();
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            using (BinaryReader labels = new BinaryReader(File.Open(labelsPath, FileMode.Open, FileAccess.Read, FileShare.Read)))
            using (BinaryReader images = new BinaryReader(File.Open(imagesPath, FileMode.Open, FileAccess.Read, FileShare.Read)))
            {
                int magicNumber = images.ReadBigInt32();
                int numberOfImages = images.ReadBigInt32();
                int width = images.ReadBigInt32();
                int height = images.ReadBigInt32();

                int magicNumber2 = labels.ReadBigInt32();
                int numberOfLabels = labels.ReadBigInt32();

                for (int i = 0; i < numberOfImages; i++)
                {
                    var bytes = images.ReadBytes(width * height);
                    var arr = new byte[height, width];

                    arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                    var image = new Image()
                    {
                        Data = arr,
                        Label = labels.ReadByte()
                    };

                    yield return image;
                }
            }
        }
    }

    public static class ImageHydratation
    {
        public static Bitmap ConvertToBitmap(byte[,] image)
        {
            Bitmap bitmap = new Bitmap(28, 28);
            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    int pixelValue = image[x, y];
                    bitmap.SetPixel(x, y, Color.FromArgb(pixelValue, pixelValue, pixelValue));
                }
            }
            return bitmap;
        }

        public static byte[,] ConvertToByteArray(Bitmap bitmap)
        {
            byte[,] result = new byte[28, 28];
            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    Color pixel = bitmap.GetPixel(x, y);
                    result[x, y] = pixel.R; // Assuming grayscale, so R = G = B
                }
            }
            return result;
        }

        public static double[] FlattenAndNormalize(byte[,] image)
        {
            double[] flattenedImage = new double[28 * 28];
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    flattenedImage[i * 28 + j] = image[i, j] / 255.0;
                }
            }
            return flattenedImage;
        }
    }

    public static class ImageAugmentation
    {
        public static void VisualizeImage(byte[,] image, string outputPath, byte label, bool save = true)
        {
            if (image.GetLength(0) != 28 || image.GetLength(1) != 28)
            {
                throw new ArgumentException("Image dimensions must be 28x28");
            }

            using (Bitmap bitmap = new Bitmap(28, 28))
            {
                for (int x = 0; x < 28; x++)
                {
                    for (int y = 0; y < 28; y++)
                    {
                        int pixelValue = image[x, y]; // Changed from image[y, x] to image[x, y]
                        Color color = Color.FromArgb(pixelValue, pixelValue, pixelValue);
                        bitmap.SetPixel(x, y, color);
                    }
                }

                // Draw the label on the image
                using (Graphics g = Graphics.FromImage(bitmap))
                {
                    g.DrawString(label.ToString(), new Font("Arial", 12), Brushes.Red, new PointF(0, 0));
                }

                outputPath = outputPath + $"-label_{label}" + ".png";

                if (save)
                {
                    bitmap.Save(outputPath, ImageFormat.Png);
                }
            }
        }

        public static byte[,] RotateImage(byte[,] image, float angle)
        {
            using (Bitmap originalBitmap = ImageHydratation.ConvertToBitmap(image))
            using (Bitmap rotatedBitmap = new Bitmap(28, 28))
            {
                using (Graphics g = Graphics.FromImage(rotatedBitmap))
                {
                    g.TranslateTransform(14, 14);
                    g.RotateTransform(angle);
                    g.TranslateTransform(-14, -14);
                    g.DrawImage(originalBitmap, new Point(0, 0));
                }

                return ImageHydratation.ConvertToByteArray(rotatedBitmap);
            }
        }

        public static byte[,] AddNoise(byte[,] image, int noiseFactor)
        {
            Random random = new Random();
            byte[,] noisyImage = new byte[28, 28];

            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    int pixelValue = image[x, y] + random.Next(-noiseFactor, noiseFactor + 1);
                    noisyImage[x, y] = (byte)Math.Clamp(pixelValue, 0, 255);
                }
            }

            return noisyImage;
        }

        public static byte[,] AdjustBrightness(byte[,] image, float factor)
        {
            byte[,] adjustedImage = new byte[28, 28];

            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    // The factor is multiplied by 255 to get the maximum value for the brightness
                    // So if factor is 1, the brightness will be the same
                    // If factor is 2, the brightness will be doubled
                    // If factor is 0.5, the brightness will be halved
                    int pixelValue = (int)(image[x, y] * factor);
                    adjustedImage[x, y] = (byte)Math.Clamp(pixelValue, 0, 255);
                }
            }

            return adjustedImage;
        }

        public static byte[,] FlipImage(byte[,] image)
        {
            byte[,] flippedImage = new byte[28, 28];

            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    flippedImage[x, y] = image[27 - x, y];
                }
            }

            return flippedImage;
        }

        public static byte[,] ApplyRandomAugmentations(byte[,] image, Random rng)
        {
            var augmentations = new List<Func<byte[,], Random, byte[,]>>
            {
                // From -10 to 10 degrees
                (byte[,] image, Random rng) => RotateImage(image, rng.Next(-10, 11)),
                // From 0.5 to 1.5
                (byte[,] image, Random rng) => AdjustBrightness(image, (float)rng.NextDouble() + 0.5f),
                // From 0 to 20
                (byte[,] image, Random rng) => AddNoise(image, rng.Next(0, 20)),
                // Flip the image
                (byte[,] image, Random rng) => FlipImage(image)
            };

            // Shuffle the order of augmentations
            augmentations = augmentations.OrderBy(x => rng.Next()).ToList();

            foreach (var augmentation in augmentations)
            {
                image = augmentation(image, rng);
            }

            return image;
        }

        public static void CreateAugmentedDataset(string outputDir, int numAugmentedImages)
        {
            Console.WriteLine("Creating augmented dataset...");
            
            // Ensure output directory exists
            Directory.CreateDirectory(outputDir);

            string imagesPath = Path.Combine(outputDir, "augmented_images.bin");
            string labelsPath = Path.Combine(outputDir, "augmented_labels.bin");

            var originalData = MnistReader.ReadTrainingData().ToList();
            Random rng = new Random();

            using (BinaryWriter imageWriter = new BinaryWriter(File.Open(imagesPath, FileMode.Create)))
            using (BinaryWriter labelWriter = new BinaryWriter(File.Open(labelsPath, FileMode.Create)))
            {
                // Write headers (similar to MNIST format)
                imageWriter.Write(BitConverter.GetBytes(0x00000803).Reverse().ToArray()); // Magic number for images
                imageWriter.Write(BitConverter.GetBytes(numAugmentedImages).Reverse().ToArray()); // Number of images
                imageWriter.Write(BitConverter.GetBytes(28).Reverse().ToArray()); // Number of rows
                imageWriter.Write(BitConverter.GetBytes(28).Reverse().ToArray()); // Number of columns

                labelWriter.Write(BitConverter.GetBytes(0x00000801).Reverse().ToArray()); // Magic number for labels
                labelWriter.Write(BitConverter.GetBytes(numAugmentedImages).Reverse().ToArray()); // Number of labels

                Parallel.For(0, numAugmentedImages, i =>
                {
                    Image originalImage = originalData[rng.Next(originalData.Count)];
                    byte[,] augmentedImage = ApplyRandomAugmentations(originalImage.Data, rng);

                    // Write augmented image
                    lock (imageWriter)
                    {
                        for (int row = 0; row < 28; row++)
                        {
                            for (int col = 0; col < 28; col++)
                            {
                                imageWriter.Write(augmentedImage[row, col]);
                            }
                        }
                    }

                    // Write label
                    lock (labelWriter)
                    {
                        labelWriter.Write(originalImage.Label);
                    }

                    if ((i + 1) % 10000 == 0 || i == numAugmentedImages - 1)
                    {
                        Console.WriteLine($"Progress: {i + 1}/{numAugmentedImages} images processed");
                    }
                });
            }

            Console.WriteLine("Augmented dataset created successfully.");
        }
    }
}
