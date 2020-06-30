# MastersProject
Hyper-Local Weather Forecasting Using Artificial Neural Networks

This project had many different looks throughout the 6 months that I worked on it.

The first term (October - Dec) mainly was used to formulate ideas for the direction of the project, and then to extract and download the data. Then, during the second term I created various artificial neural networks in order to create predictions and compare them to DarkSky results for a weather station located in Durham. 

There were many files that were created that are now obsolete and so are not included. These are mainly trying to graph the raw data that was collected in order to understand it or are older versions of the files in this respository (e.g. the LSTMRNNClass file used to be a pain to change and got a bit spaghetti like, so I recoded it so that the the neural network was in a class, with the idea that I could in the future use a genetic algorithm on it). 

Instead I have included what I believe are the most important files to the project, whereby you could take them and replicate my work with the same data.

# Short contents:
1. APICall.py - the first file I created in order to download the data that I wanted to use.
2. LSTMRNNClass.py - contains the blueprint for restructuring the data and then creating and training an LSTM RNN
3. LSTMGraphs.py - code for graphing the results from the model and comparing to other sources.
