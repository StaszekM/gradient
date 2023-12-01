import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataListLoader
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphData(pl.LightningDataModule):
    """
    Lightning Data Module for handling graph data in a PyTorch Lightning project.
    """

    def __init__(self, datamodule: Dataset):
        """
        Init GraphData class.

        Args:
            datamodule (Dataset): The dataset to be used for creating the DataLoader.
        """
        super().__init__() 

        self._dataset = self._load(datamodule) 

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training split."""
        return self._dataloader()

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation split."""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test split."""
        return self._dataloader()

    def predict_dataloader(self) -> DataLoader:
        """Returns the DataLoader for making predictions."""
        return self._dataloader()

    @property
    def num_node_features(self) -> int: 
        """Returns the number of node features in the dataset."""
        return self._dataset.num_node_features

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset."""
        return self._dataset.num_classes

    @property
    def data(self) -> Data:
        """Returns the first element of the dataset."""
        return self._dataset[0].to(device)

    @staticmethod
    def _load(datamodule: Dataset) -> Dataset:
        """Loads and prepares the dataset."""
        dataset = datamodule
        return dataset

    def _dataloader(self) -> DataLoader:
        """
        Creates and returns the DataLoader for the dataset.

        We can use the same DataLoader for all data splits, as there are masks in the Data object that we will use for selecting
        the appropriate nodes set. Moreover, we can set shuffle=False for all splits, because we have only one Data object
        (there is nothing to shuffle). Notice that we use PyTorch-Geometric's custom data loader object, because the default
        PyTorch one does not know how to collate Data objects in a batch.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        return DataListLoader(
            dataset=self._dataset,
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            )