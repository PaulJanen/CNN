from fileinput import filename
import numpy as np

class Layer:

    weightsFolderName = "weights\\"

    def __init__(self):
        self.w = None
        self.b = None

    def init(self, in_dim, initializeWeights = True):
        """
        Initializes the layer.
        Parameters
        ----------
        in_dim : int or tuple
            Shape of the input data.
        initializeWeights : bool
            Should we reinitialize weights or keep the old ones. Useful for sliding windows when we want 
            to increase layers height and width but keep trained weights.
        """
        raise NotImplementedError

    def forward(self, a_prev, training):
        """
        Propagates forward the activations.
        Parameters
        ----------
        a_prev : numpy.ndarray
            The input to this layer which corresponds to the previous layer's activations.
        training : bool
            Whether the model in which this layer is in is training.
        Returns
        -------
        numpy.ndarray
            The activations(output) of this layer.
        """
        raise NotImplementedError

    def backward(self, da):
        """
        Propagates back the gradients.
        Parameters
        ----------
        da : numpy.ndarray
            The gradients wrt the cost of this layer activations.
        Returns
        -------
        tuple
            Triplet with gradients wrt the cost of: previous layer's activations, weights and biases of this layer.
        """
        raise NotImplementedError

    def update_params(self, dw, db):
        """
        Updates parameters given their gradients.
        Parameters
        ----------
        dw : numpy.ndarray
            The gradients wrt the cost of this layer's weights.
        db : numpy.ndarray
            The gradients wrt the cost of this layer's biases.
        """
        raise NotImplementedError

    def get_params(self):
        """
        Returns
        -------
        tuple
            Trainable parameters(weights and biases) of this layer.
        """
        raise NotImplementedError

    def get_output_dim(self):
        """
        Returns
        -------
        tuple
            Shape of the ndarray layer's output.
        """
        raise NotImplementedError

    def save_params(self, fileName, w, b):
        """
        Weights and biases are saved to a file.
        Parameters
        ----------
        fileName : str
            The name of the file that will be used to load back weights and biases.
        """
        np.save(str(self.weightsFolderName + fileName+"w"), w)
        np.save(str(self.weightsFolderName + fileName+"b"), b)

    def load_params(self, fileName):
        """
        Weights and biases are saved to a file.
        Parameters
        ----------
        fileName : str
            The name of the file that will be used to load back pre-trained weights and biases.
        """

        print(fileName)
        name =  self.weightsFolderName + fileName + "w" + ".npy"
        with open(name, 'rb') as f:
            self.w = np.load(f, allow_pickle=True)
        
        name = self.weightsFolderName + fileName + "b" + ".npy"
        with open(name, 'rb') as f:
            self.b = np.load(f, allow_pickle=True)