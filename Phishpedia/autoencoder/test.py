import numpy
import torch
from data import LogoLoader
import config
from autoencoder import VAE
import os
import numpy as np
from numpy.linalg import norm
import pickle
import difflib
import time
from sklearn.decomposition import PCA
from torchvision.transforms import transforms
from PIL import Image
from config import *
from bihalf import *

vae = VAE(z_dim=64)
vae.load_state_dict(torch.load("64-vae.torch", map_location='cpu'))
vae.eval()

dataset = LogoLoader("/content/drive/MyDrive/Phishpedia/src/siamese_pedia/expand_targetlist", batch_size=1)
#dataset = LogoLoader("../src/siamese_pedia/additional/", batch_size=1)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1, shuffle=False)

pca = None
comp = 128

if os.path.exists("./logo-emb-list.pickle"):
    if not os.path.exists("./" + str(comp) + "-pca.pickle"):
        with open("./logo-emb-list.pickle", 'rb') as handle:
            print("Started")
            logos = pickle.load(handle)
            print("Loaded")
            #pca = PCA(n_components=128)
            #pca = IncrementalPCA(n_components=comp, batch_size=256)
            pca = PCA(n_components=comp)
            logo_list = numpy.array(dataset.logos)
            brand_list = dataset.brand_list
            pca_logo_list = pca.fit_transform(config.logo_feat_list)
            pca_aug_logo_list = pca.transform(logo_list)
            with open("./" + str(comp) + "-pca.pickle", 'wb') as pca_store:
                pickle.dump(pca, pca_store)
            with open("./" + str(comp) + "-pca-aug.pickle", 'wb') as pca_store:
                pickle.dump(pca_aug_logo_list, pca_store)
            with open("./" + str(comp) + "-pca-logo.pickle", 'wb') as pca_store:
                pickle.dump(pca_logo_list, pca_store)
    else:
        with open("./logo-emb-list.pickle", 'rb') as handle:
            logos = pickle.load(handle)
            logo_list = numpy.array(dataset.logos)
            brand_list = dataset.brand_list
        with open("./" + str(comp) + "-pca.pickle", 'rb') as pca_store:
            pca = pickle.load(pca_store)
        with open("./" + str(comp) + "-pca-aug.pickle", 'rb') as pca_store:
            pca_aug_logo_list = pickle.load(pca_store)
        with open("./" + str(comp) + "-pca-logo.pickle", 'rb') as pca_store:
            pca_logo_list = pickle.load(pca_store)

flat = torch.nn.Flatten()
d = []
std = numpy.std(config.logo_feat_list, 0)
mean = numpy.mean(config.logo_feat_list, 0)
std[np.isnan(std)] = 1
std[std == 0] = 1


with torch.no_grad():
    c = 0
    d = []
    prev = None
    for i, data in enumerate(config.logo_feat_list):
        #print(vae.encoder(data))
        #start = time.time()
        #print(data.shape)
        stddata = numpy.divide(data - mean, std)
        recon_images, mu, logvar = vae(torch.tensor(stddata).unsqueeze(0).unsqueeze(0))
        z = vae.reparameterize(mu, logvar)
        #hash = hash_layer(torch.round(min_max_normalization(z[0], 0, 1)))
        #ls = numpy.dot(config.logo_feat_list, recon_images[0].numpy().T / norm(recon_images[0]))
        #print("AUTO", time.time() - start)
        #start = time.time()
        #ls2 = numpy.dot(pca_logo_list, pca.transform(data[0]).T)
        #print("PCA", time.time() - start)
        #x = numpy.argmax(ls)
        #x2 = numpy.argmax(ls2)
        vec = z[0].numpy()
        leng = norm(vec)
        d.append(vec / leng)
    iden = []
    c = 0
    f = 0
    for i, data in enumerate(train_loader):
        #emb
        recon_images, mu, logvar = vae(data)
        start = time.time()
        z = vae.reparameterize(mu, logvar)
        vec = z[0].numpy() / norm(z[0].numpy())
        start = time.time()
        print(vec)
        xi = d @ vec
        xmax = numpy.argmax(xi)
        print(time.time() - start)
        start = time.time()
        x = logo_feat_list @ data[0].numpy().T
        xmax = numpy.argmax(xi)
        print(time.time() - start)
        #print(1)
        #print(xi[np.argmax(xi)])
        #print(file_name_list[np.argmax(xi)])
        if brand_converter(os.path.basename(os.path.dirname(config.file_name_list[xmax]))) != dataset.brand_list[i]:
            c += 1
        f += 1
        print(c / f)

'''
with torch.no_grad():
    for i, data in enumerate(config.logo_feat_list):
        #print(vae.encoder(data))
        #start = time.time()
        #print(data.shape)
        stddata = numpy.divide(data - mean, std)
        recon_images, mu, logvar = vae(torch.tensor(stddata).unsqueeze(0).unsqueeze(0))
        z = vae.reparameterize(mu, logvar)
        ls = numpy.dot(config.logo_feat_list, recon_images[0].numpy().T / norm(recon_images[0]))
        #print("AUTO", time.time() - start)
        #start = time.time()
        #ls2 = numpy.dot(pca_logo_list, pca.transform(data[0]).T)
        #print("PCA", time.time() - start)
        x = numpy.argmax(ls)
        #x2 = numpy.argmax(ls2)
        vec = z[0].numpy()
        leng = norm(vec)
        d.append(vec / leng)
    for i, data in enumerate(train_loader):
        with open(domain_map_path, 'rb') as handle:
            domain_map = pickle.load(handle)
        im = Image.open(r"/Volumes/GoogleDrive/Meine Ablage/LNet/Phishpedia/src/siamese_pedia/additional/Adobe/" + data)
        width, height = im.size
        # Take in entire image
        pred_boxes = [0, 0, width, height]
        #augment = transforms.Compose([
        #    transforms.RandomAffine((0, 0), translate=(0.1, 0.1), scale=(0.75, 1), fill=255, shear=(-1, 1))
        #])
        #aug = augment(im)

        brand, _, _, emb = siamese_inference(pedia_model, domain_map,
                                                 logo_feat_list, file_name_list, aug, t_s=siamese_ts,
                                                 gt_bbox=pred_boxes, getEmbedding=True)

        emb = numpy.divide(emb - mean, std)

        tensor = torch.tensor(emb)
        recon_images, mu, logvar = vae(torch.tensor(emb).unsqueeze(0).unsqueeze(0))
        z = vae.reparameterize(mu, logvar)
        vec = z[0].numpy()
        xlen = norm(vec)
        xi = numpy.dot(d, vec / xlen)
        print(data)
        #print(xi)
        print(xi[np.argmax(xi)])
        print(file_name_list[np.argmax(xi)])
        print("---------------------")'''
