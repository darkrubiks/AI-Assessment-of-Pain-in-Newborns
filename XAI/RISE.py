import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm

class RISE(nn.Module):
    """
    RISE for binary classifiers.
    - Works with models that output either:
        * a single logit/probability: shape [N, 1]
        * two logits/probabilities:   shape [N, 2] (index 1 is assumed 'positive')
    - Returns a single saliency map for the positive class.
    """
    def __init__(self, model, input_size, gpu_batch=16, device=None, pos_index=1, normalize=True):
        """
        Args:
            model: PyTorch model
            input_size: (H, W)
            gpu_batch: batch size for masking forward passes
            device: torch.device or str (e.g., 'cuda:0' or 'cpu'); if None -> auto
            pos_index: positive-class index if model has two outputs
            normalize: min-max normalize saliency to [0, 1] before returning
        """
        super().__init__()
        self.model = model.eval()  # inference mode
        self.input_size = tuple(input_size)
        self.gpu_batch = gpu_batch
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.pos_index = pos_index
        self.normalize = normalize
        self.masks = None
        self.N = 0
        self.p1 = None

    @torch.no_grad()
    def generate_masks(self, N, s, p1, savepath=None, dtype=np.float32):
        """
        Generate N random low-res Bernoulli grids (s x s), upsample & crop to HxW.

        Args:
            N: number of masks
            s: grid size (e.g., 7, 8, 10)
            p1: probability of 1s in the grid
            savepath: optional .npy to save masks
            dtype: numpy dtype for masks
        """
        H, W = self.input_size
        cell_size = np.ceil(np.array([H, W]) / s).astype(int)
        up_size = ((s + 1) * cell_size[0], (s + 1) * cell_size[1])

        # Random Bernoulli grids
        grid = (np.random.rand(N, s, s) < p1).astype(dtype)

        masks = np.empty((N, H, W), dtype=dtype)
        for i in tqdm(range(N), desc="Generating RISE masks"):
            # random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # bilinear upsample + crop
            up = resize(
                grid[i], up_size, order=1, mode="reflect", anti_aliasing=False, preserve_range=True
            ).astype(dtype)
            masks[i] = up[x:x + H, y:y + W]

        masks = masks.reshape(N, 1, H, W)  # [N,1,H,W] for broadcasting
        if savepath is not None:
            np.save(savepath, masks)

        # to tensor on device
        self.masks = torch.from_numpy(masks).to(self.device).float()
        self.N = N
        self.p1 = float(p1)

    @torch.no_grad()
    def load_masks(self, filepath):
        """
        Load precomputed masks saved by `generate_masks(..., savepath=...)`.
        """
        masks = np.load(filepath)
        self.masks = torch.from_numpy(masks).to(self.device).float()
        self.N = self.masks.shape[0]
        self.p1 = float(self.masks.mean().item())  # approximate p1 if not stored

    @torch.no_grad()
    def attribution_mask(self, x):
        """
        Compute the RISE saliency map for the positive class.

        Args:
            x: input image tensor of shape [1, C, H, W] (single image)

        Returns:
            saliency: tensor of shape [H, W], normalized to [0,1] if normalize=True
        """
        assert self.masks is not None, "Masks not initialized. Call generate_masks() or load_masks()."
        assert x.dim() == 4 and x.size(0) == 1, "Provide a single image tensor with shape [1, C, H, W]."

        x = x.to(self.device)
        N, _, H, W = self.masks.shape

        # Apply all masks to the single input: broadcasting [N,1,H,W] * [1,C,H,W]
        # -> stack: [N, C, H, W]
        stack = self.masks * x  # broadcast over C
        self.stack = stack  # for visualization/debugging

        # Forward in batches and collect positive-class scores
        scores = []
        for i in range(0, N, self.gpu_batch):
            xb = stack[i: i + self.gpu_batch]  # [B, C, H, W]
            probs = self.model.predict(xb)   # [B, 1] or [B, 2] (or probs)
            scores.append(probs)  # [B]

        scores = torch.cat(scores, dim=0)  # [N]
        print(scores)

        # Weighted sum of masks: sum_i (score_i * mask_i)
        # masks: [N,1,H,W], scores: [N] -> [N,1,1,1] for broadcasting
        sal = (self.masks * scores.view(N, 1, 1, 1)).sum(dim=0).squeeze(0)  # [H,W]

        # Normalize by expected number of ones per mask (N * p1) as in RISE
        sal = sal / (self.N * self.p1 + 1e-12)

        if self.normalize:
            # min-max to [0,1]
            mn, mx = sal.min(), sal.max()
            if (mx - mn) > 1e-12:
                sal = (sal - mn) / (mx - mn)
            else:
                sal = torch.zeros_like(sal)

        return sal  # [H, W], on self.device
