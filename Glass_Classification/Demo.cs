using Microsoft.ML;
using Microsoft.ML.Data;
using MLnetBeginner.Fake_Bank_Note_Detection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Glass_Classification
{
    internal class Demo
    {
        public static void Execute()
        {
            //RI,Na,Mg,Al,Si,K,Ca,Ba,Fe,Type

            // Create MLContext
            var context = new MLContext();

            // Data Path
            var path = "C:\\Users\\akrc2\\Downloads\\glass.csv";

            // Load data
            var data = context.Data.LoadFromTextFile<InputModel>(path: path, separatorChar: ',', hasHeader: true);

            // Prepare data & create pipeline
            var pipeline = context.Transforms.SelectColumns(
                nameof(InputModel.RI), nameof(InputModel.Na), nameof(InputModel.Mg), nameof(InputModel.Al), nameof(InputModel.Si),
                nameof(InputModel.K), nameof(InputModel.Ca), nameof(InputModel.Ba), nameof(InputModel.Fe), nameof(InputModel.Type))
                .Append(context.Transforms.Concatenate("Features", nameof(InputModel.RI), nameof(InputModel.Na), nameof(InputModel.Mg), nameof(InputModel.Al),
                nameof(InputModel.Si), nameof(InputModel.K), nameof(InputModel.Ca), nameof(InputModel.Ba), nameof(InputModel.Fe), nameof(InputModel.Type)))
                .Append(context.Transforms.Conversion.MapValueToKey("Label", nameof(InputModel.Type)))
                .Append(context.MulticlassClassification.Trainers.SdcaNonCalibrated())
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Split the data into training and testing sets
            var dataSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // Train the model
            var model = pipeline.Fit(dataSplit.TrainSet);

            // Evaluate the model
            var predictions = model.Transform(dataSplit.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");
            Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");

            // Make A Prediction
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);
            var testData = new InputModel { RI = 1.52101F, Na = 13.64F, Mg = 4.49F, Al = 1.10F, Si = 71.78F, K = 0.06F, Ca = 8.75F, Ba = 0.0F, Fe = 0.0F };
            var prediction = predictionEngine.Predict(testData);

            Console.WriteLine($"Predicted Index: {prediction.Prediction}");

            var save_path = "C:\\Users\\akrc2\\OneDrive\\Masaüstü\\ML.net - Glass Classification\\GlassClassification.zip";

            using (var fileStream = new FileStream(save_path, FileMode.Create))
            {
                context.Model.Save(model, data.Schema, fileStream);
            }
        }
    }
}
