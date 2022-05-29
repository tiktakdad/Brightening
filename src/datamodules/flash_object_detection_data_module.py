from flash.image import ObjectDetectionData


class FlashObjectDetectionData:
    def __init__(
        self,
        train_folder,
        train_ann_file,
        test_folder,
        test_ann_file,
            transform_kwargs,
            batch_size,
            pin_memory,
            num_workers,
            val_split,
    ):
        self.dataset = ObjectDetectionData.from_voc(
            train_folder=train_folder,
            train_ann_file=train_ann_file,
            test_folder=test_folder,
            test_ann_file=test_ann_file,
            transform_kwargs=transform_kwargs,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            val_split=val_split)

    @property
    def get_dataset(self):
        return self.dataset


