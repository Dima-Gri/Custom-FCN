import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        
        B, N =  target.shape

        return 1 / (B * N) * np.sum((input - target)**2) 
        return super().compute_output(input, target)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        B, N =  target.shape
        return 2 / (B * N) * (input - target) 
        return super().compute_grad_input(input, target)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        ar = self.log_softmax.compute_output(input)
        return (-1 / input.shape[0] * np.sum(ar[np.arange(len(ar)), target]))
        return super().compute_output(input, target)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        grad_output = np.zeros(input.shape)
        grad_output[np.arange(len(grad_output)), target] = -1 / input.shape[0]
        return self.log_softmax.compute_grad_input(input, grad_output)
        return super().compute_grad_input(input, target)
