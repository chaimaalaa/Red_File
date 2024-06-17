Les Conditional Generative Adversarial Networks (CGANs) sont une extension des GANs classiques, où des informations supplémentaires sont utilisées pour conditionner à la fois le générateur et le discriminateur. Cette information supplémentaire est souvent sous la forme d'étiquettes de classe ou d'autres types de données auxiliaires.

Principe des CGANs
GAN classique
Un GAN se compose de deux réseaux neuronaux en compétition :

- Le générateur (G) : Génère des échantillons de données à partir de vecteurs de bruit aléatoires.
- Le discriminateur (D) : Tente de distinguer entre les échantillons générés et les échantillons réels.
Le générateur essaie de tromper le discriminateur en produisant des échantillons de plus en plus réalistes, tandis que le discriminateur devient de plus en plus compétent pour distinguer les faux des vrais échantillons.

CGAN
Dans un CGAN, on conditionne à la fois le générateur et le discriminateur sur une information supplémentaire, généralement des étiquettes de classe. Cela permet de contrôler le type d'échantillon généré.

Architecture des CGANs
Générateur conditionnel
Le générateur prend à la fois un vecteur de bruit et une étiquette de classe comme entrée. Le bruit est généralement un vecteur aléatoire, tandis que l'étiquette de classe est souvent représentée sous forme one-hot. Ces deux vecteurs sont combinés (souvent par concaténation) avant d'être passés à travers le réseau générateur.

Discriminateur conditionnel
Le discriminateur prend une image (réelle ou générée) et une étiquette de classe comme entrée. L'étiquette de classe est souvent concaténée à l'image (après un certain traitement) avant d'être passée à travers le réseau discriminateur.
