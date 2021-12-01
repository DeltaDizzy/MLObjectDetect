using System;
using System.IO;
using Microsoft.ML;

namespace MLtest
{
    class Program
    {
        static string assetsRelativePath = @"../../../assets";
        static string assetsPath = GetAbsolutePath(assetsRelativePath);
        static string modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
        static string imagesFolder = Path.Combine(assetsPath, "images");
        static string outputFolder = Path.Combine(assetsPath, "images", "output");

        MLContext mlContext = new MLContext();

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
