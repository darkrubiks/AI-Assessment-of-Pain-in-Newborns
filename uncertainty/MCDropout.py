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

    def predict(self, x: torch.tensor, reps: int=10) -> torch.tensor:
        """
        The predict method will auto enable the dropout layers of the model,
        returning all the predictions made for the given repetition value.

        Parameters
        ----------
        x : the input of the model

        reps : the number of repetitions to generate predictions scores

        Returns
        ----------
        All predictions scores in a torch tensor
        """
        self._enable_dropout()
        predictions = torch.zeros(x.size(0), reps, dtype=torch.float, device=x.device)

        for i in range(reps):
            with torch.no_grad():
                output = self.model.predict(x)
            predictions[:, i] = output.squeeze()  # Assuming output is a single value per sample

        return predictions