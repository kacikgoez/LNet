import sys
sys.path.append('../')

from config import *
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#####################################################################################################################
# ** Step 1: Enter Layout detector, get predicted elements
# ** Step 2: Enter Siamese, siamese match a phishing target, get phishing target

# **         If Siamese report no target, Return Benign, None
# **         Else Siamese report a target, Return Phish, phishing target
#####################################################################################################################


def main(path):
    '''
    Phishdiscovery for phishpedia main script
    :param url: URL
    :param screenshot_path: path to screenshot
    :return phish_category: 0 for benign 1 for phish
    :return pred_target: None or phishing target
    :return plotvis: predicted image
    :return siamese_conf: siamese matching confidence
    '''

    path_target_list = path

    print("Eval PCA for all logos in the database")

    for dir_path in os.listdir(path_target_list):
        parent_path = os.path.join(path_target_list, dir_path + "/")
        for i in glob.glob(parent_path + "[0-9]*.png"):
            screenshot_path = str(i)
            url = "https://www.test.com"
            im = Image.open(screenshot_path)
            # Dimension of the current image
            width, height = im.size
            # Take in entire image
            pred_boxes = [[0, 0, width, height]]
            # Data Augmentation
            augment = transforms.Compose([
                transforms.RandomAffine((0,0), translate=(0.1,0.1), scale=(0.65,1), fill=255, shear=(-1.5,1.5))
                ])
            im = augment(im)
            pedia_target, matched_coord, siamese_conf = phishpedia_classifier_logo(None, logo_boxes=pred_boxes,
                                                                             domain_map_path=domain_map_path,
                                                                             model=pedia_model,
                                                                             logo_feat_list=logo_feat_list,
                                                                             file_name_list=file_name_list,
                                                                             url=url,
                                                                             shot_path=im,
                                                                             ts=siamese_ts)

            plt.imshow(im)
            plt.show()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main("../src/siamese_pedia/expand_targetlist/")
