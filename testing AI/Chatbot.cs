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
                Console.WriteLine($"predicted vector {outputVector.Length}");

                // Decode the predicted signal using GloVe embeddings
                float[] wordVector = predictedSignal.Take(embedding.VectorSize).ToArray();
                Console.WriteLine($"word vector {wordVector.Length}");
                string predictedWord = embedding.Decode(wordVector);
                predictedWords.Add(predictedWord + " ");
            }

            return predictedWords;
        }
        public void Train(List<string> inputTexts, int numEpochs = 10, int windowSize = 10)
        {
            // Encode each input text into a list of word vectors
            List<List<double[]>> inputSignalsList = new List<List<double[]>>();
            foreach (string inputText in inputTexts)
            {
                List<string> inputWords = inputText.ToLower().Split(' ').ToList();
                List<double[]> inputSignals = new List<double[]>();
                foreach (string inputWord in inputWords)
                {
                    double[] wordVector = embedding.Encode(inputWord);
                    inputSignals.Add(wordVector);
                }
                inputSignalsList.Add(inputSignals);
            }

            // Train the neural network on each word individually using a sliding window approach
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                for (int i = 0; i < inputSignalsList.Count; i++)
                {
                    List<double[]> inputSignals = inputSignalsList[i];
                    List<double[]> expectedSignals = inputSignals.Skip(1).ToList();
                    expectedSignals.Add(new double[embedding.VectorSize]); // add padding vector for last word
                    for (int j = 0; j < inputSignals.Count - 1; j++)
                    {
                        // Construct the input and expected signals for the current word
                        double[] inputSignal = new double[embedding.VectorSize * windowSize];
                        double[] expectedSignal = expectedSignals[j];
                        for (int k = 0; k < windowSize; k++)
                        {
                            int inputIndex = j + k - windowSize + 1;
                            if (inputIndex < 0)
                            {
                                // If we're at the beginning of the input text, use padding vectors
                                inputSignal[k * embedding.VectorSize] = 1;
                            }
                            else if (inputIndex >= inputSignals.Count)
                            {
                                // If we're at the end of the input text, use the last input signal
                                double[] wordVector = inputSignals[inputSignals.Count - 1];
                                Array.Copy(wordVector, 0, inputSignal, k * embedding.VectorSize, embedding.VectorSize);
                            }
                            else
                            {
                                // Otherwise, use the corresponding input signal
                                double[] wordVector = inputSignals[inputIndex];
                                Array.Copy(wordVector, 0, inputSignal, k * embedding.VectorSize, embedding.VectorSize);
                            }
                        }

                        // Train the neural network on the current word using the input and expected signals
                        float[][] inputVector = new float[][] { Array.ConvertAll(inputSignal, x => (float)x) };
                        float[][] expectedVector = new float[][] { Array.ConvertAll(expectedSignal, x => (float)x) };
                        neuralNet.Learn(inputVector, expectedVector, 1);
                    }
                }
            }
        }
    }
}
