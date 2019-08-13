using System;
using System.IO;
using System.Text;
using UnityEngine;

public static class DataExporter
{
    public static void Export(string prefix, float[] data, int width, int height)
    {
        string dirpath = Application.streamingAssetsPath + "/Data";
        string dateTimeStr = DateTime.Now.ToString("yyyyMMddHHmmss");

        StringBuilder textBuilder = new StringBuilder();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = x + width*y;
                textBuilder.Append(x + " " + y + " " + data[index] + "\n");
            }
        }

        byte[] serialized = System.Text.Encoding.ASCII.GetBytes(textBuilder.ToString());

		string filename = prefix + "_" + dateTimeStr + ".txt";
        SaveByteData(dirpath, filename, serialized);

        Debug.Log("Export has finished: " + dirpath + "/" + filename);
    }

    async static void SaveByteData(string dirpath, string filename, byte[] data)
    {
        if(!Directory.Exists(dirpath))
        {
            Directory.CreateDirectory(dirpath);
        }

        string filepath = dirpath + "/" + filename;

        using(var fs = new FileStream(filepath, FileMode.OpenOrCreate))
        {
            await fs.WriteAsync(data, 0, data.Length);
        }
    }
}