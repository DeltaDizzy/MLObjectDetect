using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLtest.YoloParser
{

    public class ImagePrediction
    {
        #region Hardcoded YOLOv4 Attributes
        private readonly float[][][] ANCHORS = new float[][][]
        {
            new float[][] { new float[] { 12, 16 }, new float[] { 19, 36 }, new float[] { 40, 28 } },
            new float[][] { new float[] { 36, 75 }, new float[] { 76, 55 }, new float[] { 72, 146 } },
            new float[][] { new float[] { 142, 110 }, new float[] { 192, 243 }, new float[] { 459, 401 } }
        };
        private readonly float[] STRIDES = new float[] { 8, 16, 32 };
        private readonly float[] XYSCALE = new float[] { 1.2f, 1.1f, 1.05f };
        private readonly int[] SHAPES = new int[] { 52, 26, 13 };

        private const int _anchorsCount = 3;
        private const float _scoreThreshold = 0.5f;
        private const float _iouThreshold = 0.5f;

        private static Color[] classColors = new Color[]
        {
            Color.Khaki,
            Color.Fuchsia,
            Color.Silver,
            Color.RoyalBlue,
            Color.Green,
            Color.DarkOrange,
            Color.Purple,
            Color.Gold,
            Color.Red,
            Color.Aquamarine,
            Color.Lime,
            Color.AliceBlue,
            Color.Sienna,
            Color.Orchid,
            Color.Tan,
            Color.LightPink,
            Color.Yellow,
            Color.HotPink,
            Color.OliveDrab,
            Color.SandyBrown,
            Color.DarkTurquoise
        };
        #endregion
        
        /// <summary>
        /// Output - Identity
        /// </summary>
        [VectorType(1, 52, 52, 3, 85)]
        [ColumnName("Identity:0")]
        public float[] Identity { get; set; }

        /// <summary>
        /// Output - Identity 1
        /// </summary>
        [VectorType(1, 26, 26, 3, 85)]
        [ColumnName("Identity_1:0")]
        public float[] Identity1 { get; set; }

        /// <summary>
        /// Output - Identity 2
        /// </summary>
        [VectorType(1, 13, 13, 3, 85)]
        [ColumnName("Identity_2:0")]
        public float[] Identity2 { get; set; }

        [ColumnName("width")]
        public float ImageWidth { get; set; }

        [ColumnName("height")]
        public float ImageHeight { get; set; }

        public IReadOnlyList<Result> GetResults(string[] categories)
        {
            var postProcessedBoundBoxes = PostProcessBoundBoxes(
                new[] { Identity, Identity1, Identity2 }, 
                categories.Length
                );
            return NMS(postProcessedBoundBoxes, categories);
        }

        private List<float> PostProcessBoundBoxes(float[][] results, int classesCount)
        {
            List<float[]> postProcesssedResults = new();

            for (int i = 0; i < results.Length; i++)
            {
                var pred = results[i];
                var outputSize = SHAPES[i];
                //https://rubikscode.net/2021/04/05/machine-learning-with-ml-net-object-detection-with-yolo/
            }
        }
    }
}
