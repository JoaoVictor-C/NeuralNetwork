using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace ImageDisplay;

public class ImageViewer
{
    public static void DisplayImage(DataHandling.Image image)
    {
        byte[] pixelData = new byte[image.size * image.size];
        for (int y = 0; y < image.size; y++)
        {
            for (int x = 0; x < image.size; x++)
            {
                int index = image.GetFlatIndex(x, y);
                pixelData[index] = (byte)(image.pixelValues[index] * 255);
            }
        }

        using (Bitmap bitmap = new Bitmap(image.size, image.size, PixelFormat.Format8bppIndexed))
        {
            BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, image.size, image.size), ImageLockMode.WriteOnly, bitmap.PixelFormat);
            Marshal.Copy(pixelData, 0, bmpData.Scan0, pixelData.Length);
            bitmap.UnlockBits(bmpData);

            ColorPalette palette = bitmap.Palette;
            for (int i = 0; i < 256; i++)
            {
                palette.Entries[i] = Color.FromArgb(i, i, i);
            }
            bitmap.Palette = palette;

            // Create a PictureBox to hold the image
            PictureBox pictureBox = new PictureBox
            {
                Dock = DockStyle.Fill,
                SizeMode = PictureBoxSizeMode.Zoom,
                Image = (Image)bitmap.Clone()
            };

            // Add the picture box to the form
            Form form = new Form
            {
                Text = $"Image Viewer - Label: {image.label}",
                ClientSize = new Size(280, 280)  // Increased size for better visibility
            };
            form.Controls.Add(pictureBox);

            // Show the form
            Application.Run(form);
        }
    }
}