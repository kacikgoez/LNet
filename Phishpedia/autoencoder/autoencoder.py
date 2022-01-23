import torch
import torch.utils.data as D
import torch.nn as nn
import torch.nn.functional as F
from data import LogoLoader
from bihalf import *
import numpy as np
from torchsummary import summary

zdimGlob = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=160):
        return input.view(input.size(0), 32, 5)


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=160, z_dim=zdimGlob):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(image_channels, 16, kernel_size=4, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 24, kernel_size=8, stride=4),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 24, kernel_size=16, stride=4),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 32, kernel_size=24, stride=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            Flatten()
            #nn.Conv1d(64, 128, kernel_size=32, stride=8),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            #Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose1d(32, 24, kernel_size=28, stride=8),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.ConvTranspose1d(24, 24, kernel_size=18, stride=4),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.ConvTranspose1d(24, 16, kernel_size=11, stride=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, image_channels, kernel_size=4, stride=2),
            nn.BatchNorm1d(image_channels),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        hash = hash_layer(min_max_normalization(torch.round(z), 0, 1))
        z = self.fc3(hash)
        return self.decoder(z), mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

if __name__ == "__main__":

    model = VAE(image_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    bs = 32
    ite = 1

    dataset = LogoLoader("/content/drive/MyDrive/Phishpedia/src/siamese_pedia/expand_targetlist", batch_size=bs)
    dataloader = D.DataLoader(dataset, batch_size=bs, shuffle=True)

    for epoch in range(epochs):
        print("Epoch " + str(ite))
        ite += 1
        for idx, (images) in enumerate(dataloader):
            recon_images, mu, logvar = model(images)
            loss, bce, kld = loss_fn(recon_images.to(device), images.to(device), mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                    epochs, loss.item(), bce.item()/bs, kld.item()/bs)

    torch.save(model.state_dict(), str(zdimGlob) + '-vae.torch')

