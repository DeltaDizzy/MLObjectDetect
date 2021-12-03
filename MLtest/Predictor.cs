using Microsoft.ML;
using MLtest.DataModel;
using MLtest.DataStructures;
using System.Drawing;

namespace MLtest
{
    public class Predictor
    {
        private MLContext _mlContext;
        private PredictionEngine<ImageNetData, ImagePrediction> _predictor;

        public Predictor(ITransformer trainedModel)
        {
            _mlContext = new MLContext();
            _predictor = _mlContext.Model.CreatePredictionEngine<ImageNetData, ImagePrediction>(trainedModel);
        }
        private ImagePrediction Predict(Bitmap image)
        {
            return _predictor.Predict(new ImageNetData()
            {
                Image = image
            });
        }
    }
}
