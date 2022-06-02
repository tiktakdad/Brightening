import PIL
import numpy as np
import torch
from PIL import ImageOps
from PIL.Image import Image as PILImage
from flash.core.serve import ModelComponent, Servable, expose, Composition
from flash.core.serve.types import Label, Image, Text, Number, BBox, Table
from flash.image import ObjectDetector
from icevision import ImgSize
from torchvision.transforms import transforms


def tensor_to_image(tensor, is_divided=False, is_trans=False):
    if is_trans:
        tensor = torch.permute(tensor, (1, 2, 0))
    if is_divided:
        tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def get_padding(max_w, max_h, width, height):
    h_padding = (max_w - height) / 2
    v_padding = (max_h - width) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

    return padding


import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, tensor):
        w, h = tensor.shape[2], tensor.shape[1]
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(tensor, padding, 0.5, 'constant')


class DetectionInference(ModelComponent):

    def __init__(self, model: Servable):
        self.model = model
        self.input_size = 640
        self.transform = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize([self.input_size, self.input_size]),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )



    @expose(
        inputs={"img": Image()},
        outputs={"prediction": Table(['minx', 'miny', 'width', 'height'])},
    )
    def detect(self, img):
        class TEMP:
            class detection:
                class_map = None

            detection = detection
            record_id = None
            original_img_size = ImgSize(width=self.input_size, height=self.input_size)
            img_size = ImgSize(width=self.input_size, height=self.input_size)
            img = None
        img = torch.squeeze(img)
        #tensor_to_image(img).save('origin.jpg')
        img = torch.permute(img, (2, 0, 1))
        img = img / 255.0
        tensor = self.transform(img)

        #tensor_to_image(tensor, True, True).save('trans.jpg')
        tensor = torch.unsqueeze(tensor, 0)
        tensor = torch.unsqueeze(tensor, 0)

        predictions = self.model([tensor, [TEMP()]])

        result = []
        for bbox in predictions[0]['bboxes']:
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            width = int(bbox['width'])
            height = int(bbox['height'])
            result.append([xmin, ymin, width, height])

        output = torch.Tensor(np.array(result))
        return output


def main():
    model = ObjectDetector.load_from_checkpoint(
        '/outputs/2022-05-30/01-49-01/object_detection_model.pt')
    comp = DetectionInference(model)
    composition = Composition(detection=comp)
    composition.serve(host='192.168.0.2', port=8000)


if __name__ == "__main__":
    main()
