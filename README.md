# MastersProject
Hyper-Local Weather Forecasting Using Artificial Neural Networks

This project had many different looks throughout the 6 months that I worked on it.

The first term (October - Dec) mainly was used to formulate ideas for the direction of the project, and then to extract and download the data.

The second term I then got to experiment with artificial neural networks.

I haven't included many of the obsolete files, as they were mainly trying to graph the data that was collected or were older versions that were more specific (e.g. the LSTMRNNClass file used to be a larger pain to change and got a bit spaghetti like, so I recoded it so that the the neural network was in a class, with the idea that I could in the future use a genetic algorithm on it). 

Instead I have included what I believe are the most important files to the project, whereby you could take them and replicate my work with the same data.

Short contents:
1. APICall.py - the first file I created in order to download the data that we wanted to use.
2. LSTMRNNClass.py - contains the blueprint for restructuring the data and then creating and training an LSTM RNN
3. LSTMGraphs.py - code for graphing the results from the model and comparing to other sources.
