import os.path

import matplotlib.pyplot
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image

from config import *
import pickle
import numpy
from sklearn.decomposition import PCA
from numpy.linalg import norm
numpy.seterr(divide='ignore', invalid='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
    Dataloder class for the autoencoder
    Provides the original dataset and an augmented one
'''

class LogoLoader(torch.utils.data.Dataset):

    def __init__(self, path, batch_size=16):
        self.batch_size=batch_size
        self.drop_last = False
        # Expects path to (expanded) logo dataset
        self. i = 1
        self.logos = []
        self.brand_list = []
        #self.transf = transforms.Compose([ transforms.ToTensor() ])
        if os.path.exists("./logo-emb-list.pickle"):
            print("Loading")
            with open("./logo-emb-list.pickle", 'rb') as handle:
                print("Loaded")
                self.loaded = pickle.load(handle)
                self.logos = []
                self.brand_list = []
                for i in self.loaded:
                    self.logos.append(i[0])
                    self.brand_list.append(i[1])
                print("Shape", numpy.array(self.logos).shape)
                #self.pca = PCA(n_components=128)
                #self.pca_logo_list = self.pca.fit_transform(self.logos)
                self.pca_logo_list = numpy.array(self.logos)
                self.std = numpy.std(self.pca_logo_list, 0)
                self.mean = numpy.mean(self.pca_logo_list, 0)
                handle.close()
            return

        with open(domain_map_path, 'rb') as handle:
            domain_map = pickle.load(handle)

        c = 1
        listdir = os.listdir(path)
        for dir_path in listdir:
            parent_path = os.path.join(path, dir_path + "/")
            # Only find PNG files with integer names
            print(str(c) + " out of " + str(len(listdir)))
            c += 1
            # All png files within folder
            globPaths = glob.glob(parent_path + "*.png")
            # Iterate, but remove homepage and login
            for i in list(filter(lambda a: ("homepage.png" not in a) and ("loginpage.png" not in a), globPaths)):
                print(i)
                screenshot_path = str(i)
                # Temporary throwaway URL
                url = "https://www.test.com"
                im = Image.open(screenshot_path)
                # Dimension of the current image
                width, height = im.size
                # Take in entire image
                pred_boxes = [0, 0, width, height]
                # Get brand of screenshot
                current_brand = brand_converter(os.path.basename(os.path.dirname(screenshot_path)))
                # Data Augmentation
                for z in range(30):
                    augment = transforms.Compose([
                        transforms.RandomAffine((0, 0), translate=(0.1, 0.1), scale=(0.65, 1), fill=255, shear=(-1, 1))
                    ])
                    aug = augment(im)
                    #predicted_brand, predicted_domain, final_sim, emb
                    brand, _, _, emb = siamese_inference(pedia_model, domain_map,
                                                         logo_feat_list, file_name_list, aug, t_s=siamese_ts,
                                                         gt_bbox=pred_boxes, getEmbedding=True)
                    print(brand, current_brand)
                    if brand == current_brand:
                        self.logos.append([emb, brand])
                    else:
                        print("Dismissed")

        with open("./logo-emb-list.pickle", "wb") as embfi:
            pickle.dump(self.logos, embfi)

    def __getitem__(self, item):
        self.std[np.isnan(self.std)] = 1
        self.std[self.std == 0] = 1
        n = numpy.divide(self.pca_logo_list[item] - self.mean, self.std)
        return torch.tensor(n).float().unsqueeze(0).to(device)

    def __len__(self):
        return len(self.logos)

    '''def  __iter__(self):
        batch = []
        for idx in self.logos:
            batch.append(torch.tensor(idx).float().unsqueeze(0))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch'''


if __name__ == "__main__":
    LogoLoader("/content/drive/MyDrive/Phishpedia/src/siamese_pedia/expand_targetlist")
    #LogoLoader("../src/siamese_pedia/expand_targetlist")