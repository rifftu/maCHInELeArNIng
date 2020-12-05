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
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            perfect = True
            for x, y in dataset.iterate_once(1):
                y = nn.as_scalar(y)
                if self.get_prediction(x) != y:
                    self.get_weights().update(x, y)
                    perfect = False
            if perfect:
                break



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # get them hyperparameters boiiiiiii

        sizes = [1, 10, 10, 10, 10, 10, 1]
        self.batch_size = 5
        self.step_size = 0.04
        self.acceptable_loss = 0.018
        # Because our function starts and ends with a number
        assert (sizes[0] == 1 and sizes[-1] == 1)

        # initialize the layers and weights
        self.layers = []
        for i in range(1, len(sizes)):
            # each layer is a tuple of (weights, bias)
            self.layers.append(nn.Parameter(sizes[i-1], sizes[i]))
            self.layers.append(nn.Parameter(1, sizes[i]))



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # relu all the layers except for the last one
        is_bias = False
        for layer in self.layers[:-2]:
            if not is_bias:
                x = nn.Linear(x, layer)
                is_bias = True
                continue
            x = nn.AddBias(x, layer)
            x = nn.ReLU(x)
            is_bias = False
        # for the last layer, no relu, just multiply and bias
        x = nn.Linear(x, self.layers[-2])
        output = nn.AddBias(x, self.layers[-1])
        return output

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
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        epoch = 0
        while True:
            losses = []
            epoch += 1

            for x, y in dataset.iterate_once(self.batch_size):
                # compute lossssss
                batch_loss = self.get_loss(x, y)
                # save loss for calculating average loss later
                losses.append(nn.as_scalar(batch_loss))
                # get them gradients
                gradients = nn.gradients(batch_loss, self.layers)
                # update the weights using gradients
                for i in range(len(self.layers)):
                    self.layers[i].update(gradients[i], -1 * self.step_size)
            # stop loop when loss is acceptable
            total_loss = sum(losses)
            # print('epoch: ' + str(epoch))
            # print('loss: ' + str(total_loss))
            if total_loss <= self.acceptable_loss:
                # print('we are ok here')
                break



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
        sizes = [784, 200, 200, 10]
        self.batch_size = 100
        self.step_size = 0.2
        self.acceptable_loss = 0.98

        # initialize the layers and weights
        self.layers = []
        for i in range(1, len(sizes)):
            # each layer is a tuple of (weights, bias)
            self.layers.append(nn.Parameter(sizes[i-1], sizes[i]))
            self.layers.append(nn.Parameter(1, sizes[i]))

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
        is_bias = False
        for layer in self.layers[:-2]:
            if not is_bias:
                x = nn.Linear(x, layer)
                is_bias = True
                continue
            x = nn.AddBias(x, layer)
            x = nn.ReLU(x)
            is_bias = False
        # for the last layer, no relu, just multiply and bias
        x = nn.Linear(x, self.layers[-2])
        output = nn.AddBias(x, self.layers[-1])
        return output

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
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        epoch = 0
        while True:
            # losses = []
            epoch += 1

            for x, y in dataset.iterate_once(self.batch_size):
                # compute lossssss
                batch_loss = self.get_loss(x, y)
                # save loss for calculating average loss later
                # losses.append(nn.as_scalar(batch_loss))
                # get them gradients
                gradients = nn.gradients(batch_loss, self.layers)
                # update the weights using gradients
                for i in range(len(self.layers)):
                    self.layers[i].update(gradients[i], -1 * self.step_size)
            # stop loop when loss is acceptable
            # avg = sum(losses)/len(losses)
            # print('epoch: ' + str(epoch))
            # print('loss: ' + str(avg))
            accuracy = dataset.get_validation_accuracy()
            # print('val acc: '+ str(accuracy))

            self.step_size = max(self.step_size * 0.95, 0.01)
            if accuracy >= self.acceptable_loss:
                # print('we are ok here')
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        sizes = [self.num_chars, 200, 200, len(self.languages)]
        self.batch_size = 200
        self.step_size = 0.2
        self.acceptable_loss = 0.83

        # initialize the layers and weights
        self.layers = []
        for i in range(1, len(sizes)):
            # each layer is a tuple of (weights, bias)
            self.layers.append(nn.Parameter(sizes[i-1], sizes[i]))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.Linear(xs[0], self.layers[0])
        for i in range(1, len(xs)):
            x = xs[i]
            h = nn.Add(nn.Linear(x, self.layers[0]), nn.Linear(h, self.layers[1]))
        return nn.Linear(h, self.layers[2])

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        epoch = 0
        while True:
            # losses = []
            epoch += 1

            for x, y in dataset.iterate_once(self.batch_size):
                # compute lossssss
                batch_loss = self.get_loss(x, y)
                # save loss for calculating average loss later
                # losses.append(nn.as_scalar(batch_loss))
                # get them gradients
                gradients = nn.gradients(batch_loss, self.layers)
                # update the weights using gradients
                for i in range(len(self.layers)):
                    self.layers[i].update(gradients[i], -1 * self.step_size)
            # stop loop when loss is acceptable
            # avg = sum(losses)/len(losses)
            # print('epoch: ' + str(epoch))
            # print('loss: ' + str(avg))
            accuracy = dataset.get_validation_accuracy()
            # print('val acc: '+ str(accuracy))

            self.step_size = max(self.step_size * 0.95, 0.01)
            if accuracy >= self.acceptable_loss:
                # print('we are ok here')
                break
