from typing import List, Optional

import flash
import hydra
from flash import DataModule
from flash.image import ObjectDetectionData, ObjectDetector
from omegaconf import DictConfig
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
from src.datamodules.flash_object_detection_data_module import FlashObjectDetectionData

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    print(ObjectDetector.available_backbones())


    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # 1. Create the DataModule

    # Init lightning datamodule
    # 2. Build the task
    log.info(f"Instantiating model <{config.model._target_}>")
    model: ObjectDetector = hydra.utils.instantiate(config.model)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: FlashObjectDetectionData.get_dataset = hydra.utils.instantiate(config.datamodule)
    """
    datamodule = ObjectDetectionData.from_coco(
        train_folder=config.datamodule.train_folder,
        train_ann_file=config.datamodule.train_ann_file,
        transform_kwargs=config.datamodule.transform_kwargs,
        batch_size=config.datamodule.batch_size,
        pin_memory=True,
        num_workers=8,
        val_split=0.1,
    )
    """


    """
    model = ObjectDetector(head=wandb.config
                           .head, backbone=wandb.config
                           .backbone,
                           num_classes=datamodule.get_train_data_module.num_classes, image_size=512)
                           """

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: flash.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    """
    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(
        max_epochs=wandb.config
            .epochs, gpus=torch.cuda.device_count(), logger=logger, callbacks=callbacks
    )
    """
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    # trainer.fit(model, datamodule=datamodule)

    # 4. Detect objects in a few images!

    # predictions = trainer.predict(model, datamodule=datamodule.get_test_data_module)
    # print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("object_detection_model.pt")

    """
    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
        """
