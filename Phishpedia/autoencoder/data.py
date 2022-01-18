import os.path

import matplotlib.pyplot
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from config import *
import pickle

'''
    Dataloder class for the autoencoder
    Provides the original dataset and an augmented one
'''

class LogoLoader(torch.utils.data.Dataset):

    logos = []

    def __init__(self, path):
        # Expects path to (expanded) logo dataset
        self. i = 1
        #self.transf = transforms.Compose([ transforms.ToTensor() ])
        if os.path.exists("./logo-emb-list.pickle"):
            with open("./logo-emb-list.pickle", 'rb') as handle:
                self.logos = pickle.load(handle)
                return

        with open(domain_map_path, 'rb') as handle:
            domain_map = pickle.load(handle)

        for dir_path in os.listdir(path):
            parent_path = os.path.join(path, dir_path + "/")
            # Only find PNG files with integer names
            for i in glob.glob(parent_path + "[0-9]*.png"):
                print(len(self.logos))
                screenshot_path = str(i)
                # Temporary throwaway URL
                url = "https://www.test.com"
                im = Image.open(screenshot_path)
                # Dimension of the current image
                width, height = im.size
                # Take in entire image
                pred_boxes = [0, 0, width, height]
                # Data Augmentation
                augment = transforms.Compose([
                    transforms.RandomAffine((0, 0), translate=(0.1, 0.1), scale=(0.65, 1), fill=255, shear=(-1.5, 1.5))
                ])
                im = augment(im)
                #predicted_brand, predicted_domain, final_sim, emb
                _, _, _, emb = siamese_inference(pedia_model, domain_map,
                                                     logo_feat_list, file_name_list, im, t_s=siamese_ts,
                                                     gt_bbox=pred_boxes, getEmbedding=True)
                self.logos.append(emb)

        with open("./logo-emb-list.pickle", "wb") as embfi:
            pickle.dump(self.logos, embfi)

    def __getitem__(self, item):
        return torch.tensor(self.logos[item]), []

    def __len__(self):
        return len(self.logos)


if __name__ == "__main__":
    LogoLoader("/content/drive/MyDrive/Phishpedia/src/siamese_pedia/expand_targetlist")
