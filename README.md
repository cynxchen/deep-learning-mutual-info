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

## Code
- `initalize.py` Build, train, and runs network. Plots can be created from running this file. Change `epochs_list = [2, 100, 250, 500, 1000, 5000]` to create plots at different epoch snapshots. Code framework based on tutorial code (Reference 3).
- `calc_info.py` Extracts information from neural network. Code based from source code (Reference 2), but I simplified it by removing unnecessary parameters.
- `mutual_information_calc.py` Calculates mutual information. Code from source code (Reference 2).
- `var_u.mat` Dataset for neural network
- `plots` Folder containing plots

## References
1. https://arxiv.org/abs/1703.00810 Research Paper
2. https://github.com/ravidziv/IDNNs Source Code
3. https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/?completed=/tensorflow-deep-neural-network-machine-learning-tutorial/ Deep Neural Network Tutorial Code
