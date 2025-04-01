import numpy as np
import torch
from torch.utils.data import Dataset

class DataNormalizer:
    def __init__(self, train_loader):
        """
        Initialize DataNormalizer by computing global mean and standard deviation
        from the training dataset.

        Parameters:
        ----------
        train_loader : DataLoader
            DataLoader for the training set to compute global normalization statistics.
        """
        # Compute global mean and std from the training data
        self.x_mean, self.x_std, self.A_mean, self.A_std, self.Y_mean, self.Y_std = self._compute_global_stats(train_loader)

    def _compute_global_stats(self, data_loader):
        """
        Compute global mean and standard deviation for x, A, and Y across the entire dataset.

        Parameters:
        ----------
        data_loader : DataLoader
            DataLoader to compute global statistics.

        Returns:
        -------
        Tuple of means and standard deviations for x, A, and Y.
        """
        x_data, A_data, Y_data = [], [], []
        for i, (x, A, Y) in enumerate(data_loader):
            x_data.append(x)
            A_data.append(A.unsqueeze(2))  # Add channel dimension for A
            Y_data.append(Y.unsqueeze(2))  # Add channel dimension for Y

        # Stack all batches together
        x_data = torch.cat(x_data, dim=0)  # Shape: [Total_samples, tlen, in_channels, H, W]
        A_data = torch.cat(A_data, dim=0)  # Shape: [Total_samples, tlen, 1, H, W]
        Y_data = torch.cat(Y_data, dim=0)  # Shape: [Total_samples, tlen, 1, H, W]

        # Compute mean and std across the entire dataset
        x_mean, x_std = x_data.mean(dim=(0, 1, 3, 4), keepdim=True), x_data.std(dim=(0, 1, 3, 4), keepdim=True) + 1e-8
        A_mean, A_std = A_data.mean(dim=(0, 1, 3, 4), keepdim=True), A_data.std(dim=(0, 1, 3, 4), keepdim=True) + 1e-8
        Y_mean, Y_std = Y_data.mean(dim=(0, 1, 3, 4), keepdim=True), Y_data.std(dim=(0, 1, 3, 4), keepdim=True) + 1e-8

        return x_mean, x_std, A_mean, A_std, Y_mean, Y_std

    def normalize_batch(self, x, A, Y):
        """
        Normalize a single batch of data using global statistics.

        Parameters:
        ----------
        x : Tensor
            Input covariates tensor (shape: [batch_size, tlen, in_channels, H, W]).
        A : Tensor
            Treatment tensor (shape: [batch_size, tlen, H, W]).
        Y : Tensor
            Outcome tensor (shape: [batch_size, tlen, H, W]).

        Returns:
        -------
        Tuple of normalized x, A, and Y tensors.
        """
        x = (x - self.x_mean) / self.x_std
        A = ((A.unsqueeze(2) - self.A_mean) / self.A_std).squeeze(2)  # Normalize and remove added channel dim
        Y = ((Y.unsqueeze(2) - self.Y_mean) / self.Y_std).squeeze(2)  # Normalize and remove added channel dim
        return x, A, Y
    
    def normalize_x(self, x):
        """
        Normalize only the x (input covariates).

        Parameters:
        ----------
        x : Tensor
            Input covariates tensor (shape: [batch_size, tlen, in_channels, H, W]).

        Returns:
        -------
        Normalized x tensor.
        """
        return (x - self.x_mean) / self.x_std

    def normalize_A(self, A):
        """
        Normalize only the A (treatment).

        Parameters:
        ----------
        A : Tensor
            Treatment tensor (shape: [batch_size, tlen, H, W]).

        Returns:
        -------
        Normalized A tensor.
        """
        return ((A.unsqueeze(2) - self.A_mean) / self.A_std).squeeze(2)

    def normalize_Y(self, Y):
        """
        Normalize only the Y (outcome).

        Parameters:
        ----------
        Y : Tensor
            Outcome tensor (shape: [batch_size, tlen, H, W]).

        Returns:
        -------
        Normalized Y tensor.
        """
        return ((Y.unsqueeze(2) - self.Y_mean) / self.Y_std).squeeze(2)

    def denormalize_Y(self, Y):
        """
        Denormalize only the Y (outcome).

        Parameters:
        ----------
        Y : Tensor
            Normalized outcome tensor (shape: [batch_size, tlen, H, W]).

        Returns:
        -------
        Denormalized Y tensor.
        """
        return (Y.unsqueeze(2) * self.Y_std + self.Y_mean).squeeze(2)

    def normalize_loader(self, data_loader):
        """
        Normalize an entire DataLoader.

        Parameters:
        ----------
        data_loader : DataLoader
            DataLoader to normalize.

        Returns:
        -------
        List of normalized batches.
        """
        normalized_data = []
        for i, (x, A, Y) in enumerate(data_loader):
            x, A, Y = self.normalize_batch(x, A, Y)
            normalized_data.append((x, A, Y))
        return normalized_data

    def normalize(self, train_loader, test_loader):
        """
        Normalize both training and testing DataLoaders using global statistics.

        Parameters:
        ----------
        train_loader : DataLoader
            DataLoader for the training set.
        test_loader : DataLoader
            DataLoader for the testing set.

        Returns:
        -------
        Tuple of normalized training and testing DataLoaders.
        """
        train_loader_normalized = self.normalize_loader(train_loader)
        test_loader_normalized = self.normalize_loader(test_loader)
        return train_loader_normalized, test_loader_normalized


class BootstrappedDataset(Dataset):
    def __init__(self, original_dataset, n_samples=None, random_seed=None):
        """
        Creates a bootstrap sample of 'original_dataset'.
        By default, n_samples = len(original_dataset).

        Parameters
        ----------
        original_dataset : Dataset
            The original PyTorch Dataset from which to sample.
        n_samples : int, optional
            Number of samples in the bootstrapped dataset.
            If None, defaults to len(original_dataset).
        random_seed : int, optional
            Random seed for reproducible sampling. If None, uses current NumPy RNG state.
        """
        self.original_dataset = original_dataset
        self.n_original = len(original_dataset)

        # If n_samples not specified, use the size of the original dataset
        if n_samples is None:
            n_samples = self.n_original
        self.n_samples = n_samples

        # Create a local random state so as not to affect global seeds
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        else:
            rng = np.random  # use the global NumPy state

        # Sample indices with replacement
        self.indices = rng.randint(low=0, high=self.n_original, size=self.n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Map the requested index to one of the sampled indices
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]

