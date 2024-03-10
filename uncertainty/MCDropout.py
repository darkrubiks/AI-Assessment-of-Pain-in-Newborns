import numpy as np
import torch


class MCDropout:
    """
    Monte-Carlo Dropout is one of the most famous way to model uncertainty in Deep-Learning.
    Considering that a Deep-Learning model was trained with Dropout layers, we can leverage
    this implementation during test phase to generate different outputs/predictions scores
    that tries to aproximate the underlying uncertainty of the input data.

    Parameters
    ----------
    model : the PyTorch model to apply MCDropout

    p : the probability of "turning off" neurons

    See Also
    --------
    Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model 
    uncertainty in deep learning." international conference on machine learning. PMLR, 2016.
    """

    def __init__(self, model: torch.nn.Module, p: float=0.5) -> None:
        self.model = model
        self.p = p

    def _enable_dropout(self):
        """
        Enables the dropout layers during inference
        """
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p = self.p
                m.train()

    def predict(self, x: torch.tensor, reps: int=10) -> np.ndarray:
        """
        The predict method will auto enable the dropout layers of the model,
        returning all the predictions made for the given repetition value.

        Parameters
        ----------
        x : the input of the model

        reps : the number of repetitions to generate predictions scores

        Returns
        ----------
        All predictions scores in a numpy array
        """
        self._enable_dropout()
        preds = np.zeros(reps)

        for i in range(reps):
            preds[i] = self.model.predict(x).detach().cpu().numpy()

        return preds