from typing import cast
from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


class HideableTQDMProgressBar(TQDMProgressBar):
    def __init__(self, hide_on_valitation=True, *args, **kwargs):
        self._hide_on_validation = hide_on_valitation

        super().__init__(*args, **kwargs)

    def init_validation_tqdm(self) -> Tqdm:
        if self._hide_on_validation:
            bar = tqdm(nrows=0, disable=True, bar_format="")
            return cast(Tqdm, bar)

        return super().init_validation_tqdm()
