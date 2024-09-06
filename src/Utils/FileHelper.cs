using System;
using System.IO;

namespace Utils;

public static class FileHelper
{
    public static byte[] ReadAllBytes(string filePath)
    {
        byte[] buffer;

        using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        {
            int length = (int)fileStream.Length;  // Get the total number of bytes
            buffer = new byte[length];            // Create a buffer to hold the bytes
            int count;                            // Actual number of bytes read
            int sum = 0;                          // Total bytes read

            // Read until EOF or the buffer is full
            while ((count = fileStream.Read(buffer, sum, length - sum)) > 0)
            {
                sum += count;  // Increment the total number of bytes read
            }
        }

        return buffer;
    }

    public static string GetFilePath(string filePath)
    {
        return Path.GetFullPath(filePath);
    }
}
