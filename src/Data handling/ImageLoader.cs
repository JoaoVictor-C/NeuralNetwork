using System;
using System.Collections.Generic;

namespace DataHandling;

public class ImageLoader
{

	int imageSize = 28;
	bool greyscale = true;
	DataFile[] dataFiles;

	public string[] labelNames;
	public Image[] images;
	public int NumImages => images.Length;
	public int InputSize => imageSize * imageSize * (greyscale ? 1 : 3);
	public int OutputSize => labelNames.Length;
	public string[] LabelNames => labelNames;

	public ImageLoader(DataFile[] dataFiles, string[] labelNames, int imageSize = 28, bool greyscale = true)
	{
		this.dataFiles = dataFiles;
		this.labelNames = labelNames;
		this.imageSize = imageSize;
		this.greyscale = greyscale;
		images = LoadImages();
	}

	public Image GetImage(int i)
	{
		return images[i];
	}

	public DataPoint[] GetAllData()
	{
		DataPoint[] allData = new DataPoint[images.Length];
		for (int i = 0; i < allData.Length; i++)
		{
			allData[i] = DataFromImage(images[i]);
		}
		return allData;
	}

	DataPoint DataFromImage(Image image)
	{
		return new DataPoint(image.pixelValues, image.label, OutputSize);
	}

	Image[] LoadImages()
	{
		List<Image> allImages = new List<Image>();

		foreach (var file in dataFiles)
		{
			Image[] images = LoadImages(file.imageFile.bytes, file.labelFile.bytes);
			allImages.AddRange(images);
		}

		return allImages.ToArray();


		Image[] LoadImages(byte[] imageData, byte[] labelData)
		{
			// Skips the first 16 bytes of the data which are metadata
			imageData = imageData.Skip(16).ToArray();
			labelData = labelData.Skip(8).ToArray();

			int numChannels = (greyscale) ? 1 : 3;
			int bytesPerImage = imageSize * imageSize * numChannels;
			int bytesPerLabel = 1;

			int numImages = imageData.Length / bytesPerImage;
			int numLabels = labelData.Length / bytesPerLabel;
			if (numImages != numLabels)
				throw new Exception($"Number of images doesn't match number of labels ({numImages} / {numLabels})");

			int dataSetSize = System.Math.Min(numImages, numLabels);
			var images = new Image[dataSetSize];

			// Scale pixel values from [0, 255] to [0, 1]
			double pixelRangeScale = 1 / 255.0;
			double[] allPixelValues = new double[imageData.Length];

			System.Threading.Tasks.Parallel.For(0, imageData.Length, (i) =>
			{
				allPixelValues[i] = imageData[i] * pixelRangeScale;
			});

			// Create images
			System.Threading.Tasks.Parallel.For(0, numImages, (imageIndex) =>
			{
				int byteOffset = imageIndex * bytesPerImage;
				double[] pixelValues = new double[bytesPerImage];
				System.Array.Copy(allPixelValues, byteOffset, pixelValues, 0, bytesPerImage);
				Image image = new Image(imageSize, greyscale, pixelValues, null, labelData[imageIndex]);
				images[imageIndex] = image;
			});

			return images;
		}


	}

	public struct DataFile
	{
		public TextAsset imageFile;
		public TextAsset labelFile;
	}

	public struct TextAsset
	{
		public byte[] bytes;
	}

}