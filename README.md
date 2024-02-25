# S5 - Convolution Neural Network for MNIST Dataset

This project contains 4 files

* model.py - It contains Convolutional Neural Network to train MNIST dataset
* utils.py - It contains utilities to train , test the model and to visualise the loss and accuracy of model for the given data
* S5.ipynb - It contains code to import required packages, device to choose, load and prepare mnist data and calls function to train, evaluate and visualise the model loss and accuracy on MNIST data.

### Convolutional Neural Network ( ***models.py*** )

* Convolutional Neural Network is defined in *models.py* file

* We have designed a Convolution Neural Network which consists of 4 2d Convolutional Layers with each having kernel of size 3 with growing channels (i.e 1 -> 32 -> 64 -> 128 -> 256 ) and 2 Fully Connected Layers.

* We have defined 2 max pooling layers which will help reduce size of image.

* We are using activation functioon as relu for all convolutions and softmax function for the final layer to predict probability of presence of each class in the given image to identify the class with max probability that determines the input image.

* Architecture looks like below.
  * Conv2D(32) -> relu -> Conv2D(64) -> MaxPool2D -> relu -> Conv2D(128)  -> relu -> Conv2D(256) -> MaxPool2D -> relu -> squeeze -> FC(50) -> relu -> FC(10) -> softmax
  * Below architecture is designed to work for input images of size 28 * 28.

### Define Functions to train and test the model with MNIST data ( ***utils.py*** )

* functions for train, test and plotting graphs to see train, test accuracy and losses of the model are provided in utils.py file.

* Steps followed as part of training model:
  * model.train() - trains model ( sets to training mode )
  * For every batch,
    * load the data and target to device ( cuda)
    * set gradients to zero ([optimizer.zero_grad()](https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)) - for every mini batch we set grad to zero before we do back propagation as pytorch accumulates gradients on subsequent backward passes. if not done, then gradients from multiple passes will be accumulated resulting in deviated output.
    * predict output for the batch of data
    * calculate cross entropy loss and sum up training loss
    * do back propagation ( accumulate gradients for each parameters which have *requires_grad* set to true) 
    * update weights([for each parameter](https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944))
    * calculate the correct predicted count
    * print the accuracy and train loss for each batch

* Steps followed as part of testing model:
  * model.eval() - set training mode to false, evaluating the model ( i.e model.train(False))
  * torch.no_grad() - disable calculation of gradients for every tensor within the block
  * for every batch,
    * load test data and target to device
    * predict output for the batch of data
    * calculate cross entropy loss and sum up the test loss
    * get the correct outputs for each batch
  * print avg test loss  and accuracy for the test dataset

### Train and test the model for MNIST data

* Define model and utility instances
* load model to device
* define [SGD optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#sgd) with model parameters, learning rate as 0.01 and momentum as 0.9 
  * SGD: random training example (or a small batch) is selected to calculate the gradient and update the model parameters. This random selection introduces randomness into the optimization process, hence the term “*stochastic*” in stochastic Gradient Descent. This method is computationally less expensive.


* define StepLR scheduler with step size as 15 and gamma as 0.1
  * Scheduler is defined to have adaptive learning rate as per gradient descent procedure.
  * usually suggested to have high learning rate at the beginning so model can explore different dimensions of predictions and then gradually decrease by the time we reach the end.
  * StepLR scheduler is used for gradual learning rate reduction.
* define cross entropy loss function
* for every epoch,
  * train the model and print the train loss and accuracy
  * evaluate model and print the test loss and accuracy
  * use scheduler to decay learning rate by gamma ( which will apply for 15 steps [lr = lr * gamma] )

### Model evaluation

We have achieved accuracy of around 99.23% on test data and avg loss around 0.02 after 20 epochs.

![Alt Text](https://github.com/das91t70/S5/tree/main/img2.png "Model loss and accuracy after 20 epochs")


### Visualise Losses and Accuracy over epochs


![Alt Text](https://github.com/das91t70/S5/tree/main/img1.png "Train, test losses and accuracy over epochs")





