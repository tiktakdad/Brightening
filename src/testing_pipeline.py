from typing import List

import flash
import hydra
from flash.image import ObjectDetectionData, ObjectDetector
from omegaconf import DictConfig
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    # if not os.path.isabs(config.ckpt_path):
    # config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

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

    # Init lightning datamodule
    datamodule = ObjectDetectionData.from_coco(
        test_folder=config.datamodule.test_folder,
        test_ann_file=config.datamodule.test_ann_file,
        transform_kwargs=config.datamodule.transform_kwargs,
        batch_size=config.datamodule.batch_size,
    )
    # datamodule: ObjectDetectionData = hydra.utils.instantiate(config.datamodule)

    # 2. Build the task
    log.info(f"Instantiating model <{config.model._target_}>")
    model: ObjectDetector = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: flash.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)
