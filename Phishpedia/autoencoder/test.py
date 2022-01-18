import numpy
import torch
from data import LogoLoader
from autoencoder import VAE
import config

dm = torch.load('./model.pth', map_location="cpu")
model = VAE()
model.load_state_dict(dm)
model.eval()


dataset = LogoLoader("/content/drive/MyDrive/Phishpedia/src/siamese_pedia/expand_targetlist")

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1, shuffle=False)

with torch.no_grad():
    last = 0
    for i, (data, _) in enumerate(train_loader):
        d,_,_ = model(data)
        ls = numpy.dot(config.logo_feat_list, torch.transpose(d[0],0,1))
        x = numpy.argmax(ls)
        print(ls[x])
