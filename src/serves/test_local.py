import numpy as np
import flash
import torch
from PIL import ImageOps
from flash.image import ObjectDetector, ObjectDetectionData
import PIL.Image as pilimg

import pandas as pd
from icevision import BaseRecord, TaskComponent, ImgSize
from icevision.core.tasks import Task
from torchvision.transforms import transforms

from src.models.flash_object_detection import FlashObjectDetector


class TEMP:
    class detection:
        class_map = None

    detection = detection
    record_id = None
    original_img_size = ImgSize(width=640, height=640)
    img_size = ImgSize(width=640, height=640)
    img = None

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = pilimg.Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def main():
    #raw_image = pilimg.open('/home/kkj/Projects/Brightening_venus/data/venus/images/test/img_000006.jpg')
    # raw_image = raw_image.resize((prediction[DataKeys.METADATA]['size'][0], raw_image.height))
    #arr = np.array(raw_image) / 255.0
    #model = FlashObjectDetector.load_from_checkpoint('/home/kkj/Projects/Brightening_venus/outputs/2022-05-30/01-49-01/object_detection_model.pt')
    model = ObjectDetector.load_from_checkpoint('/outputs/2022-05-30/01-49-01/object_detection_model.pt')
    #model = ObjectDetector.load_from_checkpoint('/home/kkj/Projects/Brightening_venus/outputs/2022-05-30/01-49-01/checkpoints/epoch_089.ckpt')
    #model.serve(input_cls=['venus'])
    test = True
    if test:
        #model.step(torch.rand(1,3, 640, 640), 0)

        class TEMP:
            class detection:
                class_map = None

            detection = detection
            record_id = None
            original_img_size = ImgSize(width=640, height=640)
            img_size = ImgSize(width=640, height=640)
            img = None

        transform = transforms.Compose(
            [transforms.Resize([640,640]),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             ]
        )
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        '''
            mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        '''
        raw_image = pilimg.open('/data/venus/images/test/img_000061.jpg')
        im = ImageOps.pad(raw_image, (640, 640), color=(124, 116, 104))
        tensor = transform(im)
        tensor = torch.unsqueeze(tensor,0)
        tensor = torch.unsqueeze(tensor, 0)



        predictions = model([tensor,[TEMP()]])
        #predictions = model.end2end()
        #predictions = model([torch.rand(1, 1, 3, 640, 640), BaseRecord()])


        print(str(predictions))
    else:
        datamodule = ObjectDetectionData.from_files(
            predict_files=[
                '/home/kkj/Projects/Brightening_venus/data/venus/images/test/img_000061.jpg',
            ],
            batch_size=1,
            transform_kwargs={"image_size": 640},
        )
        trainer = flash.Trainer()
        #predictions= model(datamodule.test_dataset)
        #model(torch.rand(640, 640))
        #model(torch.rand(640, 640))
        predictions = trainer.predict(model=model, datamodule=datamodule)
        print(predictions)

    #print(model(arr))



if __name__ == "__main__":
    main()