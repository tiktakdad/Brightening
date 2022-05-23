from typing import List, Optional

import flash
import hydra
import torch
from flash.core.data.utils import download_data
from flash.image import ObjectDetectionData, ObjectDetector
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
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
    # Dataset Credit: https://www.kaggle.com/ultralytics/coco128
    download_data(
        "https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip",
        config.data_dir,
    )

    datamodule = ObjectDetectionData.from_coco(
        train_folder=config.data_dir + "coco128/images/train2017/",
        train_ann_file=config.data_dir + "coco128/annotations/instances_train2017.json",
        val_split=0.1,
        transform_kwargs={"image_size": 512},
        batch_size=4,
    )

    # 2. Build the task
    model = ObjectDetector(
        head="efficientdet", backbone="d0", num_classes=datamodule.num_classes, image_size=512
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(
        max_epochs=10, gpus=torch.cuda.device_count(), logger=logger, callbacks=callbacks
    )
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 4. Detect objects in a few images!
    datamodule = ObjectDetectionData.from_files(
        predict_files=[
            config.data_dir + "coco128/images/train2017/000000000625.jpg",
            config.data_dir + "coco128/images/train2017/000000000626.jpg",
            config.data_dir + "coco128/images/train2017/000000000629.jpg",
        ],
        transform_kwargs={"image_size": 512},
        batch_size=4,
    )
    predictions = trainer.predict(model, datamodule=datamodule)
    print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("object_detection_model.pt")

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
