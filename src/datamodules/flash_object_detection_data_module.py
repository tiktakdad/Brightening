from typing import Tuple

from flash.core.data.utils import download_data
from flash.image import ObjectDetectionData


class FlashObjectDetectionData:
    def __init__(
        self,
        data_dir="data/",
        val_split=0.1,
        transform_kwargs={"image_size": 512},
        batch_size=4,
    ):
        self.root_data_dir = data_dir
        self.val_split = val_split
        self.transform_kwargs = transform_kwargs
        self.batch_size = batch_size
        self.prepare_data()
        self.train_datamodule, self.test_datamodule = self.make_datamodule()

    def make_datamodule(self) -> Tuple[ObjectDetectionData, ObjectDetectionData]:
        train_datamodule = ObjectDetectionData.from_coco(
            train_folder=self.root_data_dir + "coco128/images/train2017/",
            train_ann_file=self.root_data_dir + "coco128/annotations/instances_train2017.json",
            val_split=self.val_split,
            transform_kwargs={"image_size": 512},
            batch_size=4,
        )
        test_datamodule = ObjectDetectionData.from_files(
            predict_files=[
                self.root_data_dir + "coco128/images/train2017/000000000625.jpg",
                self.root_data_dir + "coco128/images/train2017/000000000626.jpg",
                self.root_data_dir + "coco128/images/train2017/000000000629.jpg",
            ],
            transform_kwargs={"image_size": 512},
            batch_size=4,
        )
        return train_datamodule, test_datamodule

    def prepare_data(self) -> None:
        # Dataset Credit: https://www.kaggle.com/ultralytics/coco128
        download_data(
            "https://www.kaggle.com/ultralytics/coco128",
            self.root_data_dir,
        )

    @property
    def get_train_data_module(self):
        return self.train_datamodule

    @property
    def get_test_data_module(self):
        return self.test_datamodule
