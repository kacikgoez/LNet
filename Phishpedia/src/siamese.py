from src.siamese_pedia.siamese_retrain.bit_pytorch.models import KNOWN_MODELS
from src.siamese_pedia.utils import brand_converter
from src.siamese_pedia.inference import siamese_inference, pred_siamese
import torch
import numpy as np
from torchvision import ops
from collections import OrderedDict
import pickle
from tqdm import tqdm
import tldextract

def phishpedia_config(num_classes:int, weights_path:str, targetlist_path:str, grayscale=False):
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

#     Prediction for targetlists
    logo_feat_list = []
    file_name_list = []

    #with open("/content/drive/MyDrive/Phishpedia/database.pickle", "rb") as db:
    with open("../database.pickle", "rb") as db:
        dataLoad = pickle.load(db)
        logo_feat_list = dataLoad[0]
        file_name_list = dataLoad[1]
        
    return model, np.asarray(logo_feat_list), np.asarray(file_name_list)

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
                          domain_map_path: str,
                          model, logo_feat_list, file_name_list, shot_path: str,
                          url: str,
                          ts: float):

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
            target_this, domain_this, this_conf = siamese_inference(model, domain_map,
                                                                    logo_feat_list, file_name_list,
                                                                    shot_path, bbox, t_s=ts, grayscale=False)

            # Domain matcher to avoid FP
            if (target_this is not None) and (tldextract.extract(url).domain not in domain_this):
                # FIXME: avoid fp due to godaddy domain parking, ignore webmail provider (ambiguous)
                if target_this == 'GoDaddy' or target_this == "Webmail Provider":
                    pred_target = None
                    this_conf = None
                else:
                    pred_target = target_this

            converted = brand_converter(pred_target)
            brand_index, counter, temp_list = to_index(converted, counter, temp_list)

            # Store data for OPS and add box to all_boxes
            if this_conf != None:
                # Add siamese_inference threshold here as well
                ops_boxes.append([min_x, min_y, max_x, max_y])
                ops_scores.append(this_conf)
                ops_indeces.append(1)

                #Append for returning
                all_boxes.append([min_x, min_y, max_x - min_x, max_y - min_y, converted, this_conf])

        # Remove overlapping bounding boxes with IoU > 10%
        indeces = []
        if len(all_boxes) > 0:
            indeces = ops.batched_nms(torch.tensor(ops_boxes), torch.tensor(ops_scores), torch.tensor(ops_indeces), float(0.05))

        #  Remove logos with low confidence after OPS
        result = []
        for i in indeces:
            box = all_boxes[i]
            if box[5] > ts or (str(box[4]) == "None" and ts > 0.6):
                result.append(box)

    return result

