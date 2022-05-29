# this is the order in which my classes will be displayed
import wandb
import PIL.Image as pilimg
import numpy as np
from flash import DataKeys
from src.callbacks.wandb_callbacks import get_wandb_logger


def wandb_bounding_boxes(trainer, predictions, labels):
    # load raw input photo
    for idx, label in enumerate(labels):
        if label is None:
            labels[idx] = 'None'

    class_id_to_label = {idx: v  for idx, v in enumerate(labels)}
    for prediction in predictions[0]:
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        filepath = prediction[DataKeys.INPUT][DataKeys.INPUT]
        raw_image = pilimg.open(filepath)
        #raw_image = raw_image.resize((prediction[DataKeys.METADATA]['size'][0], raw_image.height))
        raw_image = raw_image.resize(prediction[DataKeys.METADATA]['size'])
        #ratio_w = raw_image.width / prediction[DataKeys.METADATA]['size'][0]
        #ratio_h = raw_image.height / prediction[DataKeys.METADATA]['size'][1]
        '''
        ratio = raw_image.width / raw_image.height
        if ratio > 1.0:
            ratio_val = raw_image.width/ prediction[DataKeys.METADATA]['size'][0]
            raw_image = raw_image.resize((prediction[DataKeys.METADATA]['size'][0], int(raw_image.height / ratio_val)))
        else:
            ratio_val = raw_image.height / prediction[DataKeys.METADATA]['size'][1]
            raw_image = raw_image.resize((int(raw_image.width / ratio_val), prediction[DataKeys.METADATA]['size'][1]))
        '''

        #raw_image.thumbnail(prediction[DataKeys.METADATA]['size'], pilimg.ANTIALIAS)
        #raw_image = raw_image.thumbnail(prediction[DataKeys.METADATA]['size'], pilimg.Image.ANTIALIAS)
        all_boxes = []
        preds = prediction[DataKeys.PREDS]
        for idx in range(len(preds['bboxes'])):
            bbox = preds['bboxes'][idx]
            label = int(preds['labels'][idx])
            score = float(preds['scores'][idx])
            box_data = {"position": {
                "minX": int(bbox['xmin'] ),
                "maxX": int(bbox['xmin']  + bbox['width'] ),
                "minY": int(bbox['ymin'] ),
                "maxY": int(bbox['ymin']  + bbox['height'] )},
                "class_id": label,
                # optionally caption each box with its class and score
                "box_caption": "%s (%.3f)" % (labels[label], score),
                "domain": "pixel",
                "scores": {"score": score}}
            all_boxes.append(box_data)
        box_image = wandb.Image(raw_image,
                                boxes={"predictions": {"box_data": all_boxes, "class_labels": class_id_to_label}})
        experiment.log(
            {
                f"Images/{experiment.name}": [box_image]
            }
        )

    '''
    for prediction in predictions[DataKeys.PREDS]:

        #raw_image = load_img(filename, target_size=(log_height, log_width))
        all_boxes = []
        # plot each bounding box for this image
        for b_i, box in enumerate(prediction):
            # get coordinates and labels
            box_data = {"position" : {
              "minX" : box.xmin,
              "maxX" : box.xmax,
              "minY" : box.ymin,
              "maxY" : box.ymax},
              "class_id" : display_ids[labels[b_i]],
              # optionally caption each box with its class and score
              "box_caption" : "%s (%.3f)" % (v_labels[b_i], v_scores[b_i]),
              "domain" : "pixel",
              "scores" : { "score" : v_scores[b_i] }}
            all_boxes.append(box_data)

        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = wandb.Image(raw_image, boxes = {"predictions": {"box_data": all_boxes, "class_labels" : class_id_to_label}})

        experiment.log(
            {
                f"Images/{experiment.name}": [box_image]
            }
        )
    '''
    '''
    experiment.log(
            {
                f"Images/{experiment.name}": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(
                        val_imgs[: self.num_samples],
                        preds[: self.num_samples],
                        val_labels[: self.num_samples],
                    )
                ]
            }
        )
    '''
