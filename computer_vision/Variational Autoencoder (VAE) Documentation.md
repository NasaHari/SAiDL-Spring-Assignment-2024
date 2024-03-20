



#
#
# <a name="_aihyg732zxa"></a><a name="_zbz6i0jpadpu"></a><a name="_obwc6svkbgkj"></a>**Variational Autoencoder (VAE) Documentation**
This document provides documentation for the Variational Autoencoder (VAE) implementation using PyTorch. The VAE is a type of generative model that learns to represent high-dimensional data in a lower-dimensional latent space. This implementation focuses on training a VAE on the MNIST dataset, a collection of handwritten digits, and comparing the generation of samples from Gaussian and Normal distributions in the learned latent space.





class VAE(nn.Module):


`    `def \_\_init\_\_(self, input\_dim=784, hidden\_dim=400, latent\_dim=200, device=device):

`        `super(VAE, self).\_\_init\_\_()


`        `# encoder

`        `self.encoder = nn.Sequential(

`            `nn.Linear(input\_dim, hidden\_dim),

`            `nn.LeakyReLU(0.2),

`            `nn.Linear(hidden\_dim, latent\_dim),

`            `nn.LeakyReLU(0.2)

`            `)


`        `# latent mean and variance

`        `self.mean\_layer = nn.Linear(latent\_dim, 2)

`        `self.logvar\_layer = nn.Linear(latent\_dim, 2)


`        `# decoder

`        `self.decoder = nn.Sequential(

`            `nn.Linear(2, latent\_dim),

`            `nn.LeakyReLU(0.2),

`            `nn.Linear(latent\_dim, hidden\_dim),

`            `nn.LeakyReLU(0.2),

`            `nn.Linear(hidden\_dim, input\_dim),

`            `nn.Sigmoid()

`            `)


`    `def encode(self, x):

`        `x = self.encoder(x)

`        `mean, logvar = self.mean\_layer(x), self.logvar\_layer(x)

`        `return mean, logvar


`    `def reparameterization(self, mean, var):

`        `epsilon = torch.randn\_like(var).to(device)

`        `z = mean + var\*epsilon

`        `return z


`    `def decode(self, x):

`        `return self.decoder(

`    `def forward(self, x):

x)

`        `mean, logvar = self.encode(x)

`        `z = self.reparameterization(mean, logvar)

`        `x\_hat = self.decode(z)

`        `return x\_hat, mean, logvar



# <a name="_xczd5wnjx9yo"></a>Samples Generated

![](Aspose.Words.25a5caed-5b4d-49c1-81a0-f92c2c40b579.001.png)






![](Aspose.Words.25a5caed-5b4d-49c1-81a0-f92c2c40b579.002.png)

- **Image Quality:**
  - **Gaussian (1, 2):** Images exhibit more diverse digit styles (thicker, thinner, slanted) but might also be blurrier or distorted due to exploring a larger latent space area.
  - **Standard Normal:** Images appear sharper and less distorted, focusing on a narrower range of digit variations.
- **Diversity:**
  - **Gaussian (1, 2):** Images show a wider variety of digit appearances, including unusual or distorted digits due to the extensive sampling region.
  - **Standard Normal:** Images showcase a smaller set of digit variations, focusing on digits that are readily reconstructed from the latent space.
## <a name="_f85p13h7dt6n"></a>Initial samples of  Beta distribution

alpha=beta=1

![](Aspose.Words.25a5caed-5b4d-49c1-81a0-f92c2c40b579.003.png)

I tried changing alphas and betas i created a function to plot the difference

**Good alphas,beta[ variation + clarity]**
1\.1,1.1

1\.1, 2

4,   3

higher values of alpha (α) in the Beta distribution result in sharper features and increased distinction between foreground and background elements, while higher beta (β) values lead to more prominent or detailed features and emphasis on specific characteristics. Conversely, lower alpha values produce softer edges and potentially more variation in features, while lower beta values contribute to smoother textures and less pronounced details, resulting in a more blended or abstract style overall.


Next, I tried KL Divergence with Annealing. Annealing in the KL divergence term of VAE training gradually increases its weight, striking a balance between reconstruction accuracy and latent space regularisation to improve sample quality but it gave just grainy samples and a high loss during training i tried printing the losses during training for debugging I used linear method to change the kl weight but it id not lead to any improvement
