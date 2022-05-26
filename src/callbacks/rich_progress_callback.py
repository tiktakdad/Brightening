from pytorch_lightning.callbacks import RichProgressBar


class RichProgressBarCallback(RichProgressBar):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        print("start")
