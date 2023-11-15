import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataListLoader
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphData(pl.LightningDataModule):

    def __init__(self, datamodule):
        super().__init__() # tworzy instancję klasy LightningDataModule za pomocą metody super()

        self._dataset = self._load(datamodule) # wywołuje metodę _load(dataset_name), która wczytuje i przygotowuje dane o nazwie dataset_name

    def train_dataloader(self) -> DataLoader:
        return self._dataloader()

    def val_dataloader(self) -> DataLoader:
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        return self._dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader()

    @property
    def num_node_features(self) -> int: #zwraca liczbę cech węzłów (node features) dla zestawu danych _dataset
        return self._dataset.num_node_features

    @property
    def num_classes(self) -> int: # zwraca liczbę klas dla zestawu danych _dataset
        return self._dataset.num_classes

    @property
    def data(self) -> Data: # zwraca pierwszy element zestawu danych _dataset
        return self._dataset[0].to(device)

    @staticmethod
    def _load(datamodule) -> Dataset:
        dataset = datamodule
        return dataset

    def _dataloader(self) -> DataLoader:
        # We can use the same DataLoader for all data splits, as there are
        # masks in the Data object that we will use for selecting the
        # appropriate nodes set. Moreover, we can set shuffle=False for all
        # splits, because we have only one Data object (there is nothing
        # to shuffle). Notice that we use PyTorch-Geometric's custom data loader
        # object, because the default PyTorch one does not know how to collate
        # Data objects in a batch.
        return DataListLoader(
            dataset=self._dataset, # określa zestaw danych, który ma być używany do tworzenia obiektu DataLoader
            batch_size=1, # określa rozmiar batcha, czyli ile przykładów danych będzie przetwarzanych równocześnie w jednym kroku
            shuffle=False, # ustawienie False oznacza, że dane nie będą mieszane przed każdą epoką treningową
                           # ponieważ jest tylko jeden obiekt Data (jeden zestaw danych), nie ma potrzeby mieszania go
            num_workers=0, # określa liczbę procesów roboczych, które mają być używane do ładowania danych w tle
                           # w tym przypadku jest ustawiony na 0, co oznacza, że ładowanie danych odbywa się w głównym wątku
        )