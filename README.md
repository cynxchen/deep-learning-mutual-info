# Deep Learning & Mutual Information
Recreation of figures from 'Opening the Black Box of Deep Neural Networks via Information' (https://arxiv.org/abs/1703.00810)

The main goal of this project was to recreate Figure 2 from the paper. I used the paper's dataset and my simplified version of the source code (https://github.com/ravidziv/IDNNs) to create these plots.

Each plot represents the information plane after a set number of epochs and shows the relationship between the mutual information between the input and a hidden layer I(X;T) and the mutual information between the hidden layer and the output I(T;Y).

## Network and Plot
### Specifications
- 7 fully connected hidden layers (12-10-7-5-4-3-2)
- Activation functions: hyperbolic tangent function
- Activation function for final layer: sigmoidal function
- Used discretized outputs of layers (Divided outputs into 30 equal intervals)

### Plots
Series of information plane plots after a given number of epochs. As the number of epochs increases, I(T;Y) increases (suggesting learning) and then I(X;T) decreases (suggesting compression). Colors represent the different hidden layers. Plot shows 5 randomized networks.

![](plots/final/snapshot2.png)
![](plots/final/snapshot100.png)
![](plots/final/snapshot250.png)
![](plots/final/snapshot500.png)
![](plots/final/snapshot1000.png)
![](plots/final/snapshot5000.png)

The next plot shows the evolution of the information plane. This plots the mutual information after each epoch incrementally up to 50 epochs.

![](plots/final/Mutual_information50.png)

## References
https://arxiv.org/abs/1703.00810 Research Paper
https://github.com/ravidziv/IDNNs Source Code
https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/?completed=/tensorflow-deep-neural-network-machine-learning-tutorial/ Deep Neural Network Tutorial Code
