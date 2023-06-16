using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Glass_Classification
{
    internal class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public Single Prediction { get; set; }

        public float[] Score { get; set; }
    }
}
