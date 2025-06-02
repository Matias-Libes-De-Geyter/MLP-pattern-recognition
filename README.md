# MLP-pattern-recognition
The aim of this project was, using a very simple Fast-Forward Neural Network and Adam optimizer, to have a good accuracy on MNIST database.
I used exclusively the class of Advanced Data Analysis, and chatGPT to create writing and reading functions.

### Why C++ ?
- Firstly, I used C++ because I'm much familiar with it than Python. I was also interested to challenge myself to see if I could build a neural network from scratch, with pytorch or such libraries.
- Secondly, I plan to use the "SFML" library (with which I've done many project since highschool) to ask the user to write a number, and ask the MLP to output its guess. So the project is not yet finished ! Because for now I trained and tested the NN but not yet used it intelligently.

### Thought process
- Firstly, I created the DenseBlock class. I chose to put biases inside weight matrices, and initialise weight matrices already transposed.
- I then created the MLP class, and implemented forward pass with softmax activation at the end.
- After that, I implemented backpropagation using matrix operations and maths (I used Wikipedia's [backpropagation](https://en.wikipedia.org/wiki/Backpropagation)).
- Finally, I added Adam as detailed in the class. I used the same constants, and added a small coefficient ```1e-8``` in the expression of $$w_{ij}^{l+1}$$, such as $$\frac{1}{\sqrt(\hat{v})} \longrightarrow \frac{1}{\sqrt(\hat{v}) + 10^{-8}}$$ in case $$\sqrt(\hat{v})$$ is null.
- I tried to test it on a very simple function of a spiral with 3 classes, and I gave good results. It was how I first tested my program. Then I successfully implement the **MNIST** database.
Along the way, I had to create numerous operator transformation and other functions, that I've put in ```functions.h```.

### Hyperparameters
- I used a learning rate of $$10^{-3}$$. It seems to give good results.
- Layers are such as: [ 728, 256, 128, 10 ] with 728 the number of pixels in the flatten images, and 10 the number of classes.
- In Adam, $$\beta_m = 0.9$$ and $$\beta_v = 0.999$$.
- I used batches of 64 images, and used it on the whole dataset. Since the **MNIST** dataset has 60000 training images, we had $$60000/64 \approx 900$$ epochs. I'm not sure that's how epochs works, but for my test it worked well like this.

### Results
When ran into the whole training **MNIST** database, we get:
![output](output.png)
Here we can see the training loss and accuracy for each epochs. The latter is rising quite quickly, and then settles around a mean. Which means that the model doesn't have to be ran this long to get a good accuracy. We could maybe improve it by regularization methods.

### Opening
- I didn't implement regularization methods. No dropout, no flooding, no early stopping.
- My next move will be to implement early stopping since sometimes the loss can go down to $$0.005$$ but goes up again after. When such case is found, I could make it stop.