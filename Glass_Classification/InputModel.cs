using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Glass_Classification
{
    internal class InputModel
    {
        // RI,Na,Mg,Al,Si,K,Ca,Ba,Fe,Type
        [LoadColumn(0)]
        public float RI { get; set; }

        [LoadColumn(1)]
        public float Na { get; set; }
        
        [LoadColumn(2)]
        public float Mg { get; set; }

        [LoadColumn(3)]
        public float Al { get; set; }

        [LoadColumn(4)]
        public float Si { get; set; }

        [LoadColumn(5)]
        public float K { get; set; }

        [LoadColumn(6)]
        public float Ca { get; set; }

        [LoadColumn(7)]
        public float Ba { get; set; }

        [LoadColumn(8)]
        public float Fe { get; set; }

        [LoadColumn(9)]
        public Single Type { get; set; }
    }
}
