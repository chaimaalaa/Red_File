# Importation des bibliothèques nécessaires
import torch
import math
import random
from torch import nn
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image 
import PIL
import numpy as np
import torchvision
import pandas as pd


# Configuration des hyperparamètres de formation pour controler l'entraînement du modèle GAN : la fréquence des mises à jour du modèle, les taux d'apprentissage, et les spécificités du modèle comme la taille des images et le nombre de classes. 

epochs = 100  # Le nombre total d'itérations sur l'ensemble de données complet pour l'entraînement du modèle. // Plus le nombre d'epochs est élevé, plus le modèle a de chances d'apprendre les caractéristiques des données, mais un nombre trop élevé peut entraîner un surapprentissage (overfitting).
display_step = 200    # La fréquence à laquelle les résultats de l'entraînement tels que les pertes et les images générées seront affichés ou enregistrés, cela permet de surveiller les performances et les progrès de l'entraînement à intervalles réguliers.
batch_size = 16  #  Le nombre d'échantillons traités avant que le modèle ne mette à jour ses paramètres. // Un plus grand batch_size peut stabiliser l'entraînement mais nécessite plus de mémoire.
crit_repeats = 5  # Le nombre de fois que le discriminateur (critic) est mis à jour pour chaque mise à jour du générateur. // pour assurer que le discriminateur est suffisamment entraîné par rapport au générateur.

# Paramètres de l'optimiseur
learning_rate_G = 0.001  # Le taux d'apprentissage pour l'optimiseur du générateur, contrôle la vitesse à laquelle le générateur ajuste ses poids // Une valeur trop élevée peut rendre l'entraînement instable, tandis qu'une valeur trop basse peut ralentir l'apprentissage.
learning_rate_D = 0.0002 # Meme chose pour le discriminateur
beta_1 = 0.0  # Pour l'optimiseur Adam, ces paramètres contrôlent les taux de décroissance exponentielle pour les moments de premier et second ordre.
beta_2 = 0.9  # beta_1 est souvent mis à 0 ou une valeur basse pour WGAN, tandis que beta_2 est généralement plus élevé.
lambda_ = 10  # Coefficient pour la pénalité de gradient dans la perte de Wasserstein (utilisé pour stabiliser l'entraînement).
z_dim = 256   # La dimension du vecteur de bruit utilisé comme entrée pour le générateur.
img_channels = 3   # Le nombre de canaux dans les images générées (3 pour les images RGB).
n_images = batch_size #  Le nombre d'images générées ou traitées en un seul batch.
size = (3, 100, 100)
numClasses = 7
device = 'cuda'  # 'cuda' indique l'utilisation d'un GPU pour accélérer l'entraînement.
loss = 'W'  # Type de fonction de perte utilisée (ici 'W' pour WGAN), spécifie la méthode de calcul de la perte, influençant la stabilité et la qualité de l'entraînement.
trained = False   # Indique si le modèle a déjà été entraîné. Si True, on pourrait charger les états du modèle précédemment enregistrés pour continuer l'entraînement.

## Chargement des données d'entraînement

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # Chaque tuple (0.5, 0.5, 0.5) indique la moyenne et l'écart type pour chaque canal de couleur (Rouge, Vert, Bleu) dans les images. Chaque canal de couleur est normalisé pour avoir une moyenne de 0.5 et un écart type de 0.5 après la normalisation.

# ImageFolder : Crée un dataset à partir du répertoire data_train où les sous-répertoires sont les labels de classe.

train_ds = ImageFolder(data_train, transform = T.Compose([
    T.ToTensor(), # Convertit les images en tenseurs PyTorch // Pour convertir les images en un format compatible avec PyTorch (tenseur).
    T.Normalize(*stats)]) # Normalise les images en utilisant les moyennes et écarts types spécifiés par stats.
)

dataloader = DataLoader(      # Chargement des données
    train_ds, 
    batch_size, 
    shuffle = True, 
    num_workers = 2,   # Deux processus de travail sont utilisés pour charger les données en parallèle, accélérant ainsi le processus
    pin_memory = True  #  transférer automatiquement les données vers la mémoire GPU pour un accès plus rapide lors de l'entraînement si une GPU est utilisée.
)


# Afficher une grille d'images à partir d'un tenseur de données
def show_tensor_images(image_tensor, num_images = 25, size = (1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2 # rééchelonne les valeurs des pixels du tenseur d'image de la plage [-1, 1] (souvent utilisée dans les modèles GAN) à la plage [0, 1] (qui est la plage attendue par plt.imshow).
    image_unflat = image_tensor.detach().cpu()  # crée une copie du tenseur qui n'est pas suivi par le calcul de gradient (utile pour visualisation et évite les modifications accidentelles).// .cpu() transfère le tenseur sur le CPU s'il était sur le GPU.
    image_grid = make_grid(image_unflat[:num_images], nrow=5)  # Création d'une Grille d'Images // sélectionne les num_images premiers tenseurs d'image pour la visualisation.
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())  # permute les dimensions du tenseur de (C, H, W) (Canaux, Hauteur, Largeur) à (H, W, C) (Hauteur, Largeur, Canaux) pour correspondre au format attendu par plt.imshow.
    plt.show()


# Concaténer le bruit aléatoire avec des étiquettes de classe pour générer des images conditionnelles.
def combineVectors(v1, v2):
    return torch.cat((v1, v2), axis = 1)

# Chaque valeur entière dans labels est transformée en un vecteur binaire de taille numClasses, où toutes les valeurs sont à zéro sauf une, qui est définie à un à l'indice correspondant à la classe.
def oneHotEncode(numClasses, labels):
    return nn.functional.one_hot(labels, numClasses)
# Dans les réseaux de neurones, lors de l'entraînement d'un classificateur, il est souvent nécessaire de convertir les étiquettes de classe en format one-hot avant de les utiliser comme cibles pour la fonction de perte.


# S'assurer que les entrées du générateur et du discriminateur sont correctement dimensionnées et conditionnées, elle permet de calculer dynamiquement les dimensions d'entrée en fonction des paramètres tels que la dimension de l'espace latent, les dimensions de l'image et le nombre de classes à considérer.
def getInputDimensions(z_dim, shape, numClasses):
    generator_input_dim = z_dim + numClasses  # La combinaison du bruit et des informations de classe pour générer des images.
    discriminator_input_dim = shape[0] + numClasses   #  La combinaison de l'image et des informations de classe pour évaluer si une image est réelle ou générée.
    return generator_input_dim, discriminator_input_dim

# Cette fonction de PyTorch génère un tenseur de bruit aléatoire selon une distribution normale (gaussienne) standard. Les dimensions du tenseur seront (n_examples, z_dim), où chaque élément est un nombre aléatoire tiré de la distribution normale.
def generate_noise(n_examples, z_dim, device = 'cpu'):
    noise = torch.randn(n_examples, z_dim, device = device)
    return noise

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
    def __init__(self, z_dim = 10, img_channels = 3, hidden_dims = 64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential (
            self.make_gen_block(z_dim, hidden_dims * 4, kernel_size = 4, stride = 2, padding = 1),
            self.make_gen_block(hidden_dims * 4, hidden_dims * 8, kernel_size = 4, stride = 2, padding = 1),
            self.make_gen_block(hidden_dims * 8, hidden_dims * 16, kernel_size = 4, stride = 2, padding = 1),
            self.make_gen_block(hidden_dims * 16, hidden_dims * 32, kernel_size = 4, stride = 2, padding = 1),
            self.make_gen_block(hidden_dims * 32, hidden_dims * 16, kernel_size = 4, stride = 2, padding = 1),
            self.make_gen_block(hidden_dims * 16, hidden_dims * 4, kernel_size = 4, stride = 2, padding = 1),
            self.make_gen_block(hidden_dims * 4, img_channels, kernel_size = 4, stride = 2, padding = 15, final_layer = True),
        )
    def make_gen_block(self, input_channels, output_channels, kernel_size = 4, stride = 2, padding = 1, final_layer = False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh()
            )
    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)
    def forward(self, noise):
        noise_in = self.unsqueeze_noise(noise)
        return self.gen(noise_in)

class Critic(nn.Module):
    def __init__(self, img_channels = 3, hidden_dims = 64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential (
            self.make_crit_block(img_channels, hidden_dims),
            self.make_crit_block(hidden_dims, hidden_dims * 2),
            self.make_crit_block(hidden_dims * 2, hidden_dims * 4),
            self.make_crit_block(hidden_dims * 4, hidden_dims * 8),
            self.make_crit_block(hidden_dims * 8, hidden_dims * 16),
            self.make_crit_block(hidden_dims * 16, hidden_dims * 32, kernel_size = 1, stride = 1, padding = 0),
            self.make_crit_block(hidden_dims * 32, 1, final_layer = True)
        )
    def make_crit_block(self, input_channels, output_channels, kernel_size = 4, stride = 2, padding = 1, final_layer = False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace = True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
            )
    def forward(self, img):
        img_ = self.crit(img)
        return img_.view(len(img_), -1)
