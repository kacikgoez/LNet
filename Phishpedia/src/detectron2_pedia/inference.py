import os
from src.detectron2_pedia import detectron2_1
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np


def pred_rcnn(im, predictor):
    '''
    Perform inference for RCNN
    :param im:
    :param predictor:
    :return:
    '''
    outputs = predictor(im)

    instances = outputs['instances']
    pred_classes = instances.pred_classes  # tensor
    pred_boxes = instances.pred_boxes  # Boxes object

    # ignore this comment: 0 = button, 1 = info, 2 =  nav, 3 = pop
    logo_boxes = pred_boxes[pred_classes == 1].tensor
    input_boxes = pred_boxes[pred_classes == 0].tensor

    scores = instances.scores  # tensor
    logo_scores = scores[pred_classes == 1]
    input_scores = scores[pred_classes == 0]

    return logo_boxes, logo_scores, input_boxes, input_scores

def nav_rcnn(im, predictor):
    '''
    Perform inference for RCNN
    :param im:
    :param predictor:
    :return:
    '''
    outputs = predictor(im)

    instances = outputs['instances']
    pred_classes = instances.pred_classes  # tensor
    pred_boxes = instances.pred_boxes  # Boxes object

    # ignore this comment: 0 = button, 1 = info, 2 =  nav, 3 = pop
    button_boxes = pred_boxes[pred_classes == 0].tensor
    # Cookie boxes or other types of information
    info_boxes = pred_boxes[pred_classes == 1].tensor
    nav_boxes = pred_boxes[pred_classes == 2].tensor
    popup_boxes = pred_boxes[pred_classes == 3].tensor

    scores = instances.scores  # tensor
    button_scores = scores[pred_classes == 0]
    nav_scores = scores[pred_classes == 2]
    info_scores = scores[pred_classes == 1]
    popup_scores = scores[pred_classes == 3]

    return button_boxes, button_scores, info_scores, info_boxes, nav_boxes, nav_scores, popup_boxes, popup_scores


def config_rcnn(cfg_path, device, weights_path, conf_threshold):
    '''
    Configure weights and confidence threshold
    :param cfg_path:
    :param weights_path:
    :param conf_threshold:
    :return:
    '''
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    # uncomment if you installed detectron2 cpu version
    if device == 'cpu':
        cfg.MODEL.DEVICE = 'cpu'

    # Initialize model
    predictor = DefaultPredictor(cfg)
    return predictor


def vis(img_path, pred_boxes, logo_conf=None):
    '''
    Visualize rcnn predictions
    :param img_path: str
    :param pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :param pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :return None
    '''
    check = cv2.imread(img_path)
    pred_boxes = pred_boxes.cpu().numpy() if not isinstance(pred_boxes, np.ndarray) else pred_boxes

    # draw rectangle
    for j, box in enumerate(pred_boxes):
        cv2.rectangle(check, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (36, 255, 12), 2)
        if logo_conf is not None:
            cv2.putText(check,
                            str(logo_conf[j]),
                            (int(box[0]), int(box[1])),
                            fontScale=1,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            thickness=2,
                            color=(255,0,0))

    return check
