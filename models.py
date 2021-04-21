import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # This method will create a DotProdcut class that take the 
        #  feature values and stored weights as the input.
        # Then the dot product of this two parameters will be calculated
        #  in the DotProduct class. 
        # The method will return a DotProduct object containing the result of 
        #  dot prodcut.

        dot_product=nn.DotProduct(x, self.w)
        return dot_product

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # This method will take the feature values as the input and
        #  calculate the dot prodcut of weight and the feature values
        #  by running function run().
        # [The dot_prodcut is converted to a calar variable before 
        #   calculating the dot prodcut.]
        # If the result is non-negtive, 1 is returned
        # Otherwise, if the result is negtive, 0 is retuned.

        dot_product =self.run(x)
        res = 1 if nn.as_scalar(dot_product)>=0 else -1
        return res

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
         # This method is used to train the neural network.
        # The input is a dataset that contains the training dataset in the format of 
        #  {features} - {true label}. The variable "data[0]" is used to represent the 
        #  feature values, and the "data[1]" is used to represent the true labels.
        # The "true_count" and "false_count" are used to record the number of true prediction 
        #  and wrong prediction. The training accuracy rated is calculated by 
        #  true_count/(true_count+false_count), and the training process will not stop until 
        #  the true prediction rate reach to 100%
 
        # In the training process, the train sample with bach number equal to 1 is assigned
        #  to "train_dat", the script will iterate over the "train_dat" and compute the predicted 
        #  result (y*) by calling the function "get_predict("feature values")".  
        # If y*=y (true label), then "true_count" is increased by 1.
        # And if y* != y, the "false_count" is increased by 1 and the weight is updated as 
        #   weight = weight + direction * multiplier
        #   [The direction is represented by y*, and the multiplier is the feature vector]
        # This training process will be repeatly processed until the training accuracy reach to 100%.
    
    
        train_accuracy=0
        while train_accuracy<1.0:
            train_dat=dataset.iterate_once(1)
            true_count=0
            false_count=0
            for row in train_dat:
                # compute the prediction result y*
                predict_res = self.get_prediction(row[0])
                true_lab = nn.as_scalar(row[1])
                # compare y* with y
                if predict_res == true_lab:
                     true_count+=1
                else:
                    false_count+=1
                    # update the weight as weight = weight + direction * multiplier
                    self.w.update(row[0],nn.as_scalar(row[1]))
            train_accuracy=true_count/(false_count + true_count)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        '''
        1. set the initial trainable parameters and batch size.
        2. total size of the dataset can be evenly divisible by the batch size.
        3. b2 set to nn.Parameter(1,1) since function "run" return a batch_size × 1 node 
           that represents thr model’s prediction.
        '''
        #batch size set to 10
        self.batch_size = 10
        #set the layer size to 30
        self.layer_size = 50
        #first layer's parameter matrices and parameter vectors
        self.w1 = nn.Parameter(1, self.layer_size)
        self.b1 = nn.Parameter(1, self.layer_size)
        #sencond layer's parameter matrices and parameter vectors
        self.w2 = nn.Parameter(self.layer_size, 1)
        self.b2 = nn.Parameter(1,1)
        



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #using two-layer net function: f(x) = relu(x ⋅ W1 + b1) ⋅ W2 + b2 
        #first layer's function: relu(x ⋅ W1 + b1)
        xm_1 = nn.Linear(x, self.w1)
        adding_bias1 = nn.AddBias(xm_1,self.b1)#add a bias vector
        layer_1 = nn.ReLU(adding_bias1)

        #sencond layer's function: f(x) = relu(x ⋅ W1 + b1) ⋅ W2 + b2
        xm_2 = nn.Linear(layer_1, self.w2)
        #add a bias vector and return the perdiction
        predicted_y = nn.AddBias(xm_2, self.b2)
        return predicted_y
        #return the prediction


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        #nn.SquareLoss(a, b), where a and b both have shape batch_size × num_outputs
        #For regression problem, use nn.SquareLoss() to get batched square loss.
        loss_node = nn.SquareLoss(self.run(x), y)
        return loss_node
        #return the loss node


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #set the learning rate
        learning_rate = -0.0045
        while True:
            #iterate through dataset
            for x, y in dataset.iterate_once(self.batch_size):
                #Computes the loss
                loss = self.get_loss(x,y)
                #set the list of parameters which contains parameter matrices and parameter vectors
                parameter_list = [self.w1, self.b1, self.w2, self.b2]
                #return the gradients of the loss with respect to the parameters
                gradient_list = nn.gradients(loss, parameter_list)
                #gradient-based updates to update parameters
                self.w1.update(gradient_list[0], learning_rate)
                self.b1.update(gradient_list[1], learning_rate)
                self.w2.update(gradient_list[2], learning_rate)
                self.b2.update(gradient_list[3], learning_rate)
            #gets a loss of 0.02 or better
            #if the loss is less than 0.02, stop the loop
            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if nn.as_scalar(loss) < 0.02:
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Initialize the model parameters
        self.batch_size = 25
        self.dimension_size = 10
        self.hiddenDimensions = 200
        # the first hidden dimension's parameter matrix and parameter vector
        self.w1 = nn.Parameter(784, self.hiddenDimensions)
        self.b1 = nn.Parameter(1, self.hiddenDimensions)
        # the second hidden dimension's parameter matrix and parameter vectors
        self.w2 = nn.Parameter(self.hiddenDimensions, self.hiddenDimensions)
        self.b2 = nn.Parameter(1, self.hiddenDimensions)
        # the third dimension's parameter matrix and parameter vector
        # let "run" return an object with shape 1x10, got 1x200.
        self.w3 = nn.Parameter(self.hiddenDimensions, self.dimension_size)
        self.b3 = nn.Parameter(1, self.dimension_size)
        


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # the function of three-layer net:f(x) = relu( relu(x ⋅ W1 + b1) ⋅ W2 + b2) ⋅ W3 + b3
        # the first hidden dimension
        # first layer function: relu(x ⋅ W1 + b1)
        t1 = nn.Linear(x, self.w1)
        tb1 = nn.AddBias(t1, self.b1)
        dimension1 = nn.ReLU(tb1)
        # the second hidden dimension
        # second layer function: relu( relu(x ⋅ W1 + b1) ⋅ W2 + b2)
        t2 = nn.Linear(dimension1, self.w2)
        tb2 = nn.AddBias(t2, self.b2)
        dimension2 = nn.ReLU(tb2)
        # the third hidden dimension
        # third layer function: f(x) = relu( relu(x ⋅ W1 + b1) ⋅ W2 + b2) ⋅ W3 + b3
        t3 = nn.Linear(dimension2, self.w3)
        tb3 = nn.AddBias(t3, self.b3)
        return tb3
        #return an object with shape 1x10, got 1x200, containing predicted scores

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        #For classification problems, using nn.SoftmaxLoss() to
        # computes a batched softmax loss
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        '''
        if the learning rate is too small, the process will be very slow.
        if the learning rate is too large, the process will be converge too
        quickly and can not get the optimal solution.
        In this function, the adjusted_rate will be used to adjust the learning
        rate in the training loop by several steps:
        1. Firstly, we set a large adjusted_rate,whichis 0.3 (-0.3 in this function).
        2. Then, after every turn of the loop, adjusted_rate will be added 0.05 to
           adjust the learning rate.(adjusted_rate += 0.05)
        3. Next, when adjusted_rate is larger than the "default small learning rate"
           which is 0.005, the function will use the "default small learning rate"
           to get the optimal solution.
           (corresponding to code "learning_rate = min(-0.005, adjusted_rate)")
        By using the adjusted_rate, the function can be efficient to get the optimal solution
        '''
        adjusted_rate = -0.3
        while True:
            #iterate through dataset
            for x, y in dataset.iterate_once(self.batch_size):
                #Computes the loss
                loss = self.get_loss(x, y)
                #set the list of parameters which contains parameter matrices and parameter vectors
                parameters = ([self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                ##return the gradients of the loss with respect to the parameters
                gradients = nn.gradients(loss, parameters)
                learning_rate = min(-0.005, adjusted_rate)
                # updates gradients
                self.w1.update(gradients[0], learning_rate)
                self.w2.update(gradients[1], learning_rate)
                self.w3.update(gradients[2], learning_rate)
                self.b1.update(gradients[3], learning_rate)
                self.b2.update(gradients[4], learning_rate)
                self.b3.update(gradients[5], learning_rate)
            adjusted_rate += 0.05#adjusting the learning rate
            #set a slightly higher stopping threshold on validation accuracy which is 98%
            #when the validation accuracy us higher than 98%, the training terminates
            if dataset.get_validation_accuracy() >= 0.98:
                return

