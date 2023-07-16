using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Chatbot
    {
        private readonly GloVeEmbedding embedding;
        private readonly NeuralNet neuralNet;
        private readonly int numOutputWords;

        public Chatbot(GloVeEmbedding embedding, NeuralNet neuralNet, int numOutputWords = 5)
        {
            this.embedding = embedding;
            this.neuralNet = neuralNet;
            this.numOutputWords = numOutputWords;
        }
        public List<string> PredictNextWords(string inputText, int numWords = 5, int windowSize = 10)
        {
            // Split the input text into words
            List<string> inputWords = inputText.ToLower().Split(' ').ToList();

            // Encode each word in the input window separately using GloVe embeddings
            List<double[]> inputSignals = new List<double[]>();
            // Reduce the window size if there are not enough words in the input text
            if (inputWords.Count < windowSize)
            {
                windowSize = inputWords.Count;
            }

            // Encode each word in the input window separately using GloVe embeddings
            for (int i = inputWords.Count - windowSize; i < inputWords.Count; i++)
            {
                double[] wordVector = embedding.Encode(inputWords[i]);
                inputSignals.Add(wordVector);
            }
            // Predict each word individually based on the input signals
            List<string> predictedWords = new List<string>();
            for (int i = 0; i < numWords; i++)
            {
                // Construct the input signal for the current word
                double[] inputSignal = new double[embedding.VectorSize * windowSize];
                for (int j = 0; j < windowSize; j++)
                {
                    int inputIndex = i + j - windowSize + 1;
                    if (inputIndex < 0)
                    {
                        // If we're at the beginning of the input text, use padding vectors
                        inputSignal[j * embedding.VectorSize] = 1;
                    }
                    else if (inputIndex >= inputSignals.Count)
                    {
                        // If we're at the end of the input text, use the last input signal
                        double[] sw = inputSignals[inputSignals.Count - 1];
                        Array.Copy(sw, 0, inputSignal, j * embedding.VectorSize, embedding.VectorSize);
                    }
                    else
                    {
                        // Otherwise, use the corresponding input signal
                        double[] sw = inputSignals[inputIndex];
                        Array.Copy(sw, 0, inputSignal, j * embedding.VectorSize, embedding.VectorSize);
                    }
                }

                // Predict the current word based on the input signal
                float[][] inputVector = new float[][] { Array.ConvertAll(inputSignal, x => (float)x) };
                float[][] outputVector = neuralNet.Activate(inputVector);
                float[] predictedSignal = Array.ConvertAll(outputVector[0], x => (float)x);
                Console.WriteLine($"predicted vector {outputVector[0].Length}");

                // Decode the predicted signal using GloVe embeddings
                float[] wordVector = predictedSignal.Take(embedding.VectorSize).ToArray();
                Console.WriteLine($"word vector {wordVector.Length}");
                string predictedWord = embedding.Decode(wordVector);
                predictedWords.Add(predictedWord + " ");
            }

            return predictedWords;
        }
        public void Train(string document, int epochs, int windowSize = 10)
        {
            // Split the document into words
            List<string> words = document.Split(' ').ToList();

            // Generate training data by sliding a window of fixed size over the words and using the next words as targets
            List<double[]> inputSignals = new List<double[]>();
            List<double[]> expectedSignals = new List<double[]>();
            for (int i = 0; i < words.Count - windowSize - numOutputWords + 1; i++)
            {
                List<string> inputWords = words.GetRange(i, windowSize);
                double[] inputSignal = new double[embedding.VectorSize * windowSize];
                for (int j = 0; j < windowSize; j++)
                {
                    double[] wordVector = embedding.Encode(inputWords[j]);
                    Array.Copy(wordVector, 0, inputSignal, j * embedding.VectorSize, embedding.VectorSize);
                }
                inputSignals.Add(inputSignal);

                List<string> expectedWords = words.GetRange(i + windowSize, numOutputWords);
                double[] expectedSignal = new double[embedding.VectorSize * numOutputWords];
                for (int j = 0; j < numOutputWords; j++)
                {
                    double[] wordVector = embedding.Encode(expectedWords[j]);
                    Array.Copy(wordVector, 0, expectedSignal, j * embedding.VectorSize, embedding.VectorSize);
                }
                expectedSignals.Add(expectedSignal);
            }

            // Train the neural network with the generated training data for the specified number of epochs
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Console.WriteLine($"Epoch {epoch}/{epochs}");
                for (int i = 0; i < inputSignals.Count; i++)
                {
                    float[][] inputSignal = new float[][] { Array.ConvertAll(inputSignals[i], x => (float)x) };
                    float[][] expectedSignal = new float[][] { Array.ConvertAll(expectedSignals[i], x => (float)x) };
                    neuralNet.Learn(inputSignal, expectedSignal, 1);
                }
            }
        }
    }
}
