# deimos_task_3b
Code for classification

This code is used to do classification on the cifar-10 datatset which is commonly used to test machine learning algorithms. It consists of 32*32 pixel images for easy processing.
First we load the datatset using load while setting the input to utf-8 encoding.
Then we load the testing and training datatsets. Then we pre-process the datasets by normalizing pixel values to lie between 0 and 1.
The output layer consists of 10 classes so we can classify into any of the 10.
Then we make a sequential CNN model with multiple layers with varying number of neurons with relu activation. Relu is used for nonlinearity in our classification as we want to end up with either true or false not a numerical value.
Then we compile the model with Adam optimizer with learning rate 0.001. We can use a low learning rate as we will be doing 50 epochs of 64 trials each.
We use early stopping to get the best weights at end of each epoch.

Then we test the accuracy of the model by using evaluate.
The predicted and true classes are obtained by finding the index of the maximum value in the one-hot encoded vectors.
We then create a confusion matrix showing true positives, false positives, true negatives and false negatives for each of the classes. The denser the diagonal(representing true positives) is better is out model.
This is how we can perform classification on a model and then test the accuracy of our model.
