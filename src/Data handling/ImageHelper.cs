using System.Collections;
using System.Collections.Generic;

namespace DataHandling;

public static class ImageHelper
{


	public static double[] ReadImage(byte[] imageData, int byteOffset, int imageSize, bool flip = false)
	{

		double[] pixelValues = new double[imageSize * imageSize];
		for (int pixelIndex = 0; pixelIndex < pixelValues.Length; pixelIndex++)
		{

			if (flip)
			{
				int x = pixelIndex % imageSize;
				int y = pixelIndex / imageSize;
				int flippedIndex = (imageSize - y - 1) * imageSize + (x);
				pixelValues[pixelIndex] = imageData[byteOffset + flippedIndex] / 255.0;
			}
			else
			{
				pixelValues[pixelIndex] = imageData[byteOffset + pixelIndex] / 255.0;
			}
		}

		return pixelValues;
	}

	public static byte[] ImagesToBytes(Image[] images)
	{
		List<byte> allBytes = new List<byte>();
		foreach (var image in images)
		{
			allBytes.AddRange(ImageToBytes(image));
		}
		return allBytes.ToArray();
	}

	public static byte[] ImageToBytes(Image image)
	{
		byte[] bytes = new byte[image.numPixels];
		for (int i = 0; i < bytes.Length; i++)
		{
			bytes[i] = (byte)(image.pixelValues[i] * 255);
		}
		return bytes;
	}


}