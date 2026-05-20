using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DisPix
{
    public partial class MainWindow : Window
    {
        private string? _selectedFilePath;
        private string _modelPath = "Models/generator.onnx";
        private bool _isModelAvailable = false;

        public MainWindow()
        {
            InitializeComponent();
            CheckModelStatus();
        }

        private void CheckModelStatus()
        {
            _isModelAvailable = File.Exists(_modelPath);
            EngineStatusText.Text = _isModelAvailable 
                ? "AI Neural Network (generator.onnx active)" 
                : "Mathematical Local Perturbation (Standard)";
        }

        private void OnSliderChanged(object? sender, RangeBaseValueChangedEventArgs e)
        {
            if (StrengthText != null)
            {
                StrengthText.Text = $"{(int)e.NewValue}%";
            }
        }

        private async void OnBrowseClicked(object? sender, RoutedEventArgs e)
        {
            var files = await this.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = "Select Image to Protect",
                AllowMultiple = false,
                FileTypeFilter = new[] { FilePickerFileTypes.ImageAll }
            });

            if (files != null && files.Count > 0)
            {
                _selectedFilePath = files[0].Path.LocalPath;
                PathTextBox.Text = _selectedFilePath;
                StatusText.Text = "Image loaded.";
            }
        }

        private async void OnProtectClicked(object? sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedFilePath) || !File.Exists(_selectedFilePath))
            {
                StatusText.Text = "Please select a valid image first.";
                return;
            }

            ProtectButton.IsEnabled = false;
            StatusText.Text = "Processing... Please wait.";
            double strength = StrengthSlider.Value / 100.0;

            try
            {
                string outputPath = GenerateOutputPath(_selectedFilePath);

                await Task.Run(() =>
                {
                    if (_isModelAvailable)
                    {
                        ApplyAiNoise(_selectedFilePath, outputPath, strength);
                    }
                    else
                    {
                        ApplyMathematicalNoise(_selectedFilePath, outputPath, strength);
                    }
                });

                StatusText.Text = $"Protected file saved as: {Path.GetFileName(outputPath)}";
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
            }
            finally
            {
                ProtectButton.IsEnabled = true;
            }
        }

        private string GenerateOutputPath(string inputPath)
        {
            string dir = Path.GetDirectoryName(inputPath) ?? string.Empty;
            string fileName = Path.GetFileNameWithoutExtension(inputPath);
            string ext = Path.GetExtension(inputPath);
            return Path.Combine(dir, $"{fileName}_dispix{ext}");
        }

        private void ApplyMathematicalNoise(string inputPath, string outputPath, double strength)
        {
            using var image = Image.Load<Rgb24>(inputPath);
            var random = new Random();

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image[x, y];
                    double luminance = (0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B) / 255.0;
                    double noiseScale = strength * (luminance > 0.5 ? (1.0 - luminance) : luminance) * 255.0;

                    int noiseR = (int)((random.NextDouble() * 2 - 1) * noiseScale);
                    int noiseG = (int)((random.NextDouble() * 2 - 1) * noiseScale);
                    int noiseB = (int)((random.NextDouble() * 2 - 1) * noiseScale);

                    byte r = (byte)Math.Clamp(pixel.R + noiseR, 0, 255);
                    byte g = (byte)Math.Clamp(pixel.G + noiseG, 0, 255);
                    byte b = (byte)Math.Clamp(pixel.B + noiseB, 0, 255);

                    image[x, y] = new Rgb24(r, g, b);
                }
            }

            image.Save(outputPath);
        }

        private void ApplyAiNoise(string inputPath, string outputPath, double strength)
        {
            using var image = Image.Load<Rgb24>(inputPath);
            using var session = new InferenceSession(_modelPath);

            int modelWidth = 256; 
            int modelHeight = 256;

            using var resizedImg = image.Clone(ctx => ctx.Resize(modelWidth, modelHeight));
            var inputTensor = new DenseTensor<float>(new[] { 1, 3, modelHeight, modelWidth });

            for (int y = 0; y < modelHeight; y++)
            {
                for (int x = 0; x < modelWidth; x++)
                {
                    var pixel = resizedImg[x, y];
                    inputTensor[0, 0, y, x] = pixel.R / 255f;
                    inputTensor[0, 1, y, x] = pixel.G / 255f;
                    inputTensor[0, 2, y, x] = pixel.B / 255f;
                }
            }

            var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
            using var results = session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int tx = (int)((float)x / image.Width * modelWidth);
                    int ty = (int)((float)y / image.Height * modelHeight);

                    var pixel = image[x, y];

                    float noiseR = outputTensor[0, 0, ty, tx] * (float)strength * 255f;
                    float noiseG = outputTensor[0, 1, ty, tx] * (float)strength * 255f;
                    float noiseB = outputTensor[0, 2, ty, tx] * (float)strength * 255f;

                    byte r = (byte)Math.Clamp(pixel.R + noiseR, 0, 255);
                    byte g = (byte)Math.Clamp(pixel.G + noiseG, 0, 255);
                    byte b = (byte)Math.Clamp(pixel.B + noiseB, 0, 255);

                    image[x, y] = new Rgb24(r, g, b);
                }
            }

            image.Save(outputPath);
        }
    }
}