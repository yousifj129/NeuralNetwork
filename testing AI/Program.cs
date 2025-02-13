﻿using System;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    internal class Program
    {
        
        static void Main(string[] args)
        {
            // Load the GloVe embedding and create a new neural network
            GloVeEmbedding embedding = new GloVeEmbedding("glove.6B.50d.txt");
            NeuralNet neuralNet = new NeuralNet.Builder().SetNeuronsInputLayer(50).SetNeuronsForLayers(100, 50).SetActivationFunc(ActivationFunc.ReLU).Build();

            // Create a new chatbot instance
            Chatbot chatbot = new Chatbot(embedding, neuralNet);

            // Train the chatbot on a sample document
            string[] document = new string[]{
                "hello what are you doing man",
                "are you ok?"
                };
            chatbot.Train(document.ToList(), 100, 3);
            
            // Predict the next words based on an input text
            string inputText = "The quick brown";
            int numWords = 3;
            var predictedWords = chatbot.PredictNextWords(inputText, numWords);

            // Print the predicted words
            Console.WriteLine("Predicted words:");
            foreach (string predictedWord in predictedWords)
            {
                Console.Write(predictedWord);
            }
            Console.ReadLine();
        }
    }
}
