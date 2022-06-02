# service.py
import typing as t

import numpy as np
import torch
from PIL import ImageOps
from PIL.Image import Image as PILImage

import bentoml
from bentoml.io import Image, Text
from bentoml.io import NumpyNdarray
from flash.image import ObjectDetector
from icevision import ImgSize
from torchvision.transforms import transforms
import PIL.Image as pilimg

detection_runner = bentoml.pytorch.load_runner(
    "detector:t2kttgw74oyduiyp",
    name="detector_runner",
)

svc = bentoml.Service(
    name="detector_demo",
    runners=[
        detection_runner,
    ],
)
#model = ObjectDetector.load_from_checkpoint('/outputs/2022-05-30/01-49-01/object_detection_model.pt')


async def predict_image(f: PILImage) -> str:
    assert isinstance(f, PILImage)

    class TEMP:
        class detection:
            class_map = None

        detection = detection
        record_id = None
        original_img_size = ImgSize(width=640, height=640)
        img_size = ImgSize(width=640, height=640)
        img = None

    transform = transforms.Compose(
        [transforms.Resize([640, 640]),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         ]
    )
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    '''
        mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    '''
    im = ImageOps.pad(f, (640, 640), color=(124, 116, 104))
    #raw_image = pilimg.open('/home/kkj/Projects/Brightening_venus/data/venus/images/test/img_000061.jpg')
    #im = ImageOps.pad(raw_image, (640, 640), color=(124, 116, 104))
    tensor = transform(im)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = torch.unsqueeze(tensor, 0)
    detection_runner.run

    # predictions = model()

    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension
    # arr = np.expand_dims(arr, 0).astype("float32")
    # output_tensor = await detection_runner.async_run([tensor, [TEMP()]])
    predictions = model([tensor, [TEMP()]])
    return str(predictions)
