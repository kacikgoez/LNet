from src.siamese_pedia.siamese_retrain.bit_pytorch.models import KNOWN_MODELS
from src.siamese_pedia.utils import brand_converter
from src.siamese_pedia.inference import siamese_inference, pred_siamese, lnet_siamese_inference
import torch
import numpy as np
from torchvision import ops
from collections import OrderedDict
import pickle
from tqdm import tqdm
import tldextract
import os
from copy import copy
from PIL import Image
import time

def phishpedia_config(num_classes:int, weights_path:str, targetlist_path:str, keras_pipeline, grayscale=False):
    '''
    Load phishpedia configurations
    :param num_classes: number of protected brands
    :param weights_path: siamese weights
    :param targetlist_path: targetlist folder
    :param grayscale: convert logo to grayscale or not, default is RGB
    :return model: siamese model
    :return logo_feat_list: targetlist embeddings
    :return file_name_list: targetlist paths
    '''
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location=device)
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k.split('module.')[1]
        new_state_dict[name]=v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    logo_feat_list = []
    word_list = []
    file_name_list = []
    removed_list = None

    # Load embeddings that were saved prior
    if os.path.exists(r"../database.pickle"):
        with open("../database.pickle", "rb") as db:
            dataLoad = pickle.load(db)
            logo_feat_list = dataLoad[0]
            file_name_list = dataLoad[1]
            word_list = dataLoad[2]
            removed_list = copy(file_name_list)
            db.close()

    # Check for new logos added and store them
    for target in tqdm(os.listdir(targetlist_path)):
        if target.startswith('.'): # skip hidden files
            continue
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if logo_path.endswith('.png') or logo_path.endswith('.jpeg') or logo_path.endswith('.jpg') or logo_path.endswith('.PNG') \
                                          or logo_path.endswith('.JPG') or logo_path.endswith('.JPEG'):
                if logo_path.startswith('loginpage') or logo_path.startswith('homepage'): # skip homepage/loginpage
                    continue
                # Check if file is already in existing stored db
                completePath = os.path.join(r'./src/siamese_pedia/expand_targetlist/', target, logo_path)
                if completePath not in file_name_list:
                    logo_feat_list.append(pred_siamese(img=os.path.join(targetlist_path, target, logo_path),
                                                   model=model, grayscale=grayscale))
                    file_name_list.append(completePath)
                    image_file = Image.open("." + completePath)
                    image_file = image_file.resize((int(image_file.width * (100 / image_file.height)), 100), resample=Image.BOX)
                    ocr = keras_pipeline.recognize([np.asarray(image_file.convert("RGB"))])
                    list_words = []
                    for i in ocr:
                        for j in i:
                            list_words.append(j[0])
                    word_list.append(list_words)
                else:
                    # Remove existing files to find out old files in DB
                    if removed_list is not None:
                        removed_list.remove(completePath)

    if removed_list is not None:
        for i in removed_list:
            index = file_name_list.index(i)
            print("Note: removed file from DB: " + str(index))
            del file_name_list[index]
            del logo_feat_list[index]
            del word_list[index]

    with open("../database.pickle", "wb") as db:
        data = [logo_feat_list, file_name_list, word_list]
        pickle.dump(data, db)
        db.close()

    return model, np.asarray(logo_feat_list), np.asarray(file_name_list), word_list

# Original Phishpedia implementation
def phishpedia_classifier_logo(logo_boxes,
                          domain_map_path: str,
                          model, logo_feat_list, file_name_list, shot_path: str,
                          url: str,
                          ts: float):
    '''
    Run siamese
    :param logo_boxes: torch.Tensor/np.ndarray Nx4 logo box coords
    :param domain_map_path: path to domain map dict
    :param model: siamese model
    :param logo_feat_list: targetlist embeddings
    :param file_name_list: targetlist paths
    :param shot_path: path to image
    :param url: url
    :param ts: siamese threshold
    :return pred_target
    :return coord: coordinate for matched logo
    '''
    # targetlist domain list
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    print('number of logo boxes:', len(logo_boxes))
    matched_coord = None
    siamese_conf = None

    # run logo matcher
    pred_target = None
    if len(logo_boxes) > 0:
        # siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            target_this, domain_this, this_conf = siamese_inference(model, domain_map,
                                                                    logo_feat_list, file_name_list,
                                                                    shot_path, bbox, t_s=ts, grayscale=False)

            # print(target_this, domain_this, this_conf)
            # domain matcher to avoid FP
            if (target_this is not None) and (tldextract.extract(url).domain not in domain_this):
                # FIXME: avoid fp due to godaddy domain parking, ignore webmail provider (ambiguous)
                if target_this == 'GoDaddy' or target_this == "Webmail Provider":
                    target_this = None  # ignore the prediction
                    this_conf = None
                pred_target = target_this
                matched_coord = coord
                siamese_conf = this_conf
                break  # break if target is matched
            if i >= 2:  # only look at top-2 logo
                break

    return brand_converter(pred_target), matched_coord, siamese_conf

def to_index(brand, counter, list):
    if str(brand) in list:
        return list[str(brand)], counter, list
    list[str(brand)] = counter
    return counter, (counter + 1), list

def lnet_phishpedia_classifier_logo(logo_boxes,
                          logo_conf,
                          domain_map_path: str,
                          model, logo_feat_list, file_name_list, word_list, shot_path: str,
                          url: str,
                          ts: float,
                          keras_pipeline,
                          debug = False):

    '''
    Altered Version for Layout Network
    :param logo_boxes: torch.Tensor/np.ndarray Nx4 logo box coords
    :param domain_map_path: path to domain map dict
    :param model: siamese model
    :param logo_feat_list: targetlist embeddings
    :param file_name_list: targetlist paths
    :param shot_path: path to image
    :param url: url
    :param ts: siamese threshold
    :return pred_target
    :return coord: coordinate for matched logo
    '''
    # targetlist domain list
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    print('number of logo boxes:', len(logo_boxes))

    result = []
    res_ops = []

    # run logo matcher
    pred_target = None
    if len(logo_boxes) > 0:
        # Current list of logos required for PyTorch's ops function
        ops_boxes = []
        # required for PyTorch's ops function
        ops_scores = []
        # Indeces required for PyTorch's ops function
        temp_list = {}
        counter = 0
        ops_indeces = []
        # Store all bounding boxes to be returned
        all_boxes = []

        # Siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            start = time.time()
            target_this, domain_this, this_conf = lnet_siamese_inference(model, domain_map,
                                                                    logo_feat_list, file_name_list, word_list,
                                                                    shot_path, bbox, keras_pipeline=keras_pipeline, t_s=ts, grayscale=False)
            print("Inference", time.time() - start)
            # Domain matcher to avoid FP
            if (target_this is not None) and (tldextract.extract(url).domain not in domain_this):
                # FIXME: avoid fp due to godaddy domain parking, ignore webmail provider (ambiguous)
                if target_this == 'GoDaddy' or target_this == "Webmail Provider":
                    pred_target = None
                    this_conf = None
                else:
                    pred_target = target_this
            else:
                pred_target = target_this

            converted = brand_converter(pred_target)
            print(converted)
            brand_index, counter, temp_list = to_index(converted, counter, temp_list)

            # Store data for OPS and add box to all_boxes
            if this_conf != None:
                # Add siamese_inference threshold here as well
                ops_boxes.append([min_x, min_y, max_x, max_y])
                # Make sure non-logos have lower priority to not overshadow, yet preserve their order
                if str(converted) == "None":
                    ops_scores.append(float(ts - 0.01 - (1/logo_conf[i]) * 0.01))
                else:
                    ops_scores.append(float(this_conf))
                # Take collision of all boxes into consideration
                ops_indeces.append(1)
                #Append for returning
                all_boxes.append([int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y), converted, this_conf, logo_conf[i]])
        # Remove overlapping bounding boxes with IoU > 10%
        indeces = []
        print(all_boxes)
        if len(all_boxes) > 0:
            indeces = ops.batched_nms(torch.tensor(ops_boxes), torch.tensor(ops_scores), torch.tensor(ops_indeces), float(0.1))
        result = []
        for i in indeces:
            box = all_boxes[i]
            print(box)
            # Depending on non-logo or logo, decrease logoness score
            if (str(box[4]) != "None" and box[5] > ts) or (str(box[4]) == "None" and box[5] > 0.5):
                result.append(box)
                res_ops.append(ops_boxes[i])
    if debug == True:
        return result, res_ops
    return result, []

