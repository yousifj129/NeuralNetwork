using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class GloVeEmbedding
    {
        private readonly Dictionary<string, float[]> wordVectors;

        public int VectorSize { get; }

        public GloVeEmbedding(string filePath, int vectorSize = 50)
        {
            wordVectors = new Dictionary<string, float[]>();
            VectorSize = vectorSize;

            // Load the GloVe word vectors from the file
            using (StreamReader reader = new StreamReader(filePath))
            {
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();
                    string[] tokens = line.Split(' ');
                    string word = tokens[0];
                    float[] vector = tokens.Skip(1).Select(float.Parse).ToArray();
                    wordVectors[word] = vector;
                }
            }
        }

        public double[] Encode(string text)
        {
            // Encode the text as the average of its word vectors
            string[] words = text.Split(' ');
            double[] vectorSum = new double[VectorSize];
            int count = 0;
            foreach (string word in words)
            {
                if (wordVectors.TryGetValue(word, out float[] vector))
                {
                    for (int i = 0; i < VectorSize; i++)
                    {
                        vectorSum[i] += vector[i];
                    }
                    count++;
                }
            }
            if (count > 0)
            {
                for (int i = 0; i < VectorSize; i++)
                {
                    vectorSum[i] /= count;
                }
                return vectorSum;
            }
            else
            {
                return Enumerable.Repeat(0.0, VectorSize).ToArray();
            }
        }

        public string Decode(float[] vector)
        {
            // Decode the vector as the closest word vector
            string closestWord = null;
            float closestDistance = float.PositiveInfinity;
            foreach (KeyValuePair<string, float[]> entry in wordVectors)
            {
                float distance = EuclideanDistance(vector, entry.Value);
                if (distance < closestDistance)
                {
                    closestWord = entry.Key;
                    closestDistance = distance;
                }
            }
            return closestWord;
        }

        private static float EuclideanDistance(float[] a, float[] b)
        {
            if(a.Length != b.Length)
            {
                Console.WriteLine($"error euclidean {a.Length}, {b.Length}");
                throw new Exception("scammmm");
                return 10;
            }
            float sum = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return ((float)Math.Sqrt(sum));
        }
    }
}
