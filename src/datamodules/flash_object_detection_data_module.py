from flash.core.data.utils import download_data
from flash.image import ObjectDetectionData


class FlashObjectDetectionData(ObjectDetectionData):
    def __init__(
        self,
        data_dir="data/",
        dataset_type="coco128",
        val_split=0.1,
        transform_kwargs={"image_size": 512},
        batch_size=4,
    ):
        super().__init__(val_split, batch_size)
        self.root_data_dir = data_dir
        self.val_split = val_split
        self.transform_kwargs = transform_kwargs
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.prepare_data()

    def make_datamodule(self):
        datamodule = None
        if self.dataset_type == "coco":
            datamodule = ObjectDetectionData.from_coco(
                train_folder=self.root_data_dir + "coco/images/train2017/",
                train_ann_file=self.root_data_dir + "coco/annotations/instances_train2017.json",
                test_folder=self.root_data_dir + "coco/images/val2017/",
                test_ann_file=self.root_data_dir + "coco/annotations/instances_val2017.json",
                val_split=self.val_split,
                transform_kwargs=self.transform_kwargs,
                batch_size=4,
            )
        elif self.dataset_type == "coco128":
            datamodule = ObjectDetectionData.from_coco(
                train_folder=self.root_data_dir + "coco128/images/train2017/",
                train_ann_file=self.root_data_dir + "coco128/annotations/instances_train2017.json",
                val_split=self.val_split,
                transform_kwargs=self.transform_kwargs,
                batch_size=4,
            )

        return datamodule

    def prepare_data(self) -> None:
        # Dataset Credit: https://www.kaggle.com/ultralytics/coco128
        download_data(
            "https://www.kaggle.com/ultralytics/coco128",
            self.root_data_dir,
        )
