import bentoml
from flash.image import ObjectDetector
import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
from src.datamodules.flash_object_detection_data_module import FlashObjectDetectionData

dotenv.load_dotenv(override=True)


@hydra.main(config_path="../../configs/", config_name="deploy.yaml")
def main(config: DictConfig):
    f_data: FlashObjectDetectionData = hydra.utils.instantiate(config.datamodule)
    datamodule = f_data.get_dataset
    model: ObjectDetector = hydra.utils.instantiate(config.model, num_classes=datamodule.num_classes)
    tag = bentoml.pytorch.save(
        "detector",
        model
    )


if __name__ == "__main__":
    main()