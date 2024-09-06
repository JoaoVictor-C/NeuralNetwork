namespace DataHandling;
public class Image
{
	public readonly int size;
	public readonly int numPixels;
	public readonly bool greyscale;
	public readonly double[] pixelValues;
	public readonly byte[] image;
	public readonly int label;

	public Image(int size, bool greyscale, double[] pixelValues, byte[] image, int label)
	{
		this.size = size;
		this.numPixels = size * size;
		this.greyscale = greyscale;
		this.pixelValues = pixelValues;
		this.image = image;
		this.label = label;
	}

	public int GetFlatIndex(int x, int y)
	{
		return y * size + x;
	}

	public double Sample(double u, double v)
	{
		u = System.Math.Max(System.Math.Min(1, u), 0);
		v = System.Math.Max(System.Math.Min(1, v), 0);

		double texX = u * (size - 1);
		double texY = v * (size - 1);

		int indexLeft = (int)(texX);
		int indexBottom = (int)(texY);
		int indexRight = System.Math.Min(indexLeft + 1, size - 1);
		int indexTop = System.Math.Min(indexBottom + 1, size - 1);

		double blendX = texX - indexLeft;
		double blendY = texY - indexBottom;

		double bottomLeft = pixelValues[GetFlatIndex(indexLeft, indexBottom)];
		double bottomRight = pixelValues[GetFlatIndex(indexRight, indexBottom)];
		double topLeft = pixelValues[GetFlatIndex(indexLeft, indexTop)];
		double topRight = pixelValues[GetFlatIndex(indexRight, indexTop)];

		double valueBottom = bottomLeft + (bottomRight - bottomLeft) * blendX;
		double valueTop = topLeft + (topRight - topLeft) * blendX;
		double interpolatedValue = valueBottom + (valueTop - valueBottom) * blendY;
		return interpolatedValue;
	}
}