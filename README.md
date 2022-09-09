# Projet Data Science/ML : reconnaissance de fruits en fonction d'une image

Ce projet à été réalisé par M.Bruneau Oscar et moi-même dans le cadre d'une UE de Data Science en L3 informatique, Université Paris-Saclay (grade : 20/20).
Ce projet à pour but de développer un classifieur Naif Baeysien pour la reconnaissance d'image de fruit à partir d'une image. Pour cela, nous avons imaginé une pipeline pour choisir la meilleur représentation possible de nos données d'images notée $X \in R^{n,d}$ avec $n$ le nombre d'images du dataset et $d$ le nombre de dimension. C'est cette partie qui va être analyser, quelle dimension allons nous choisir pour nos données étant donné que notre classifieur naif bayésien repose sur une modélisation de la loi de Bernoulli. Nos features(dimensions) doivent valoir soit 0 soit 1, conformément à ce choix de modélisation.

Nous avons donc choisi d'entreprendre trois méthodes de pre processing sur nos données :

- La traduction des images vers leur densité avec un histogramme de nuance de gris 
- La divisions des images en patch avec clusturisation et application d'un algorithme KMeans pour associer a chaque image un vecteur de clusters et si un patch de l'image appartient à un cluster alors la valeur associé sera 1 et 0 sinon.
- La binarisation des pixels noir ou blanc

![alt text](https://github.com/jdufou1/fruit_recognition/blob/main/img/pipeline.png)

## Récupération des données

Pour ce projet, les données d'images ont été collectées depuis un dépot Kaggle (https://www.kaggle.com/datasets/moltean/fruits) qui dispose d'une base de données de plus 90 000 fruits et légumes de 131 types différents. Pour pouvoir recopier les résultats de ce projet, vous aurez besoin de téléchager la base de 1Gb et ensuite placer le dossier Training à la racine du projet pour que fonctions de chargement des images puissent fonctionner.
Les fruits qui auront été retenu pour ce projet sont (20 classes): Banane,Fraise,Framboise,Clementine,Poire,Nectarine,Orange,Lychee,Kiwi,Mirtille,Cerise,Abricot,Citron,Melon,Tangelo,Plum,Peche,Kaki,Mangue. Avec un total de 9651 images.

Enfin, d'après les observations, les images sont de bonnes qualitée car il n'y a aucun bruit dans les images et elles sont toutes de dimensions 50x50x3.

## Pre processing : Binarisation des pixels

Ce qui est important avec ce choix de modélisation, c'est qu'il faut que nos données soit binaire, c'est à dire 0/1, car c'est le critère de l'expérience de Bernouilli. Donc, dans cette partie nous allons étudier les performances d'un pre processing sur les images en transformant nos images du format RVB vers une binarisation noir/blanc. ``./implementation/projet_pixels.ipynb``

<div style="display:flex;">
    <img src="https://github.com/jdufou1/L3_IAS_fruit_recognition_project/blob/main/img/abricot.PNG" alt="drawing" width="20%"/>
    <img src="https://github.com/jdufou1/L3_IAS_fruit_recognition_project/blob/main/img/abricot_nb.PNG" alt="drawing" width="20%"/>
</div>


| Données  | performance          |
| :---------------: |:---------------:| 
| train  | 84.16% |
| test  | 83.84%  |

Nombre de paramètres à apprendre : 50x50x20 = 50 000

## Pre processing : Densité sur les nuances de gris

Ce qui est important avec ce choix de modélisation, c'est qu'il faut que nos données soit binaire, c'est à dire 0/1, car c'est le critère de l'expérience de Bernouilli. Donc, dans cette partie nous allons étudier les performance d'un pre processing sur les images en transformant nos images en nuance de gris puis en récupérant la densité de couleur en fonction de deux hyper paramètres qui sont le nombre de bins et la valeur seuil $\alpha$ à partir duquel les valeurs sont binarisées et enfin nous appliquons un masque à ce dernier pour binariser les entrées.``./implementation/projet_densite.ipynb``

<div style="display:flex;">
    <img src="https://github.com/jdufou1/L3_IAS_fruit_recognition_project/blob/main/img/abricot_binarise.PNG" alt="drawing" width="30%"/>
    <img src="https://github.com/jdufou1/L3_IAS_fruit_recognition_project/blob/main/img/nuance_gris_densite_abricot.PNG" alt="drawing" width="45%"/>
</div>

| Données  | performance          |
| :---------------: |:---------------:| 
| train  |80.12% |
| test  | 79.75%          |

Après plusieurs tests, nous constatons que les meilleurs paramètres sont : 
- $\alpha = 0.01$
- $nbbins = 200$

Nombre de paramètres à apprendre : 200 x 20 = 4 000

## Pre processing : Patch et Clustering

Dans cette partie nous allons étudier les performance d'un pre processing sur la découpage d'image en patch. Ces derniers seront ensuite mit dans un Kmeans afin de calculer les clusters. Enfin les images seront traduites par un vecteur de taille égale au nombre de clusters et qui contiendra 1 si un patch appartient à ce cluster.
``./implementation/projet_clusters.ipynb``

<div style="display:flex;">
    <img src="https://github.com/jdufou1/L3_IAS_fruit_recognition_project/blob/main/img/repartition_cluster.PNG" alt="drawing" width="43%"/>
    <img src="https://github.com/jdufou1/L3_IAS_fruit_recognition_project/blob/main/img/erreur_clusters.PNG" alt="drawing" width="45%"/>
</div>

| Données  | performance          |
| :---------------: |:---------------:| 
| train  | 84.57% |
| test  | 84.80%          |

Nous avons retenu **120** clusters pour **4** patchs. Ce choix à été réalisé en implémentant un algorithme de cross validation.

Nombre de paramètres à apprendre : 120 x 20 = 2 400

## Modèle du bayesien naïf : modélisation avec la loi de Bernoulli

Pour contruire notre fonction de classification, on va représenter nos données sous forme binaire afin de pouvoir effectuer une expérience de Bernoulli dessus. C'est à dire que chaque dimension d'un exemple doit valoir soit 0 soit 1 afin qu'on puisse calculer ce type de probabilité :
<p align="center">
    <img src="https://latex.codecogs.com/png.image?\large&space;\dpi{110}\bg{black}P(X_d&space;=&space;x_d&space;|&space;\theta&space;,&space;y)&space;=&space;p_i^{x_d}*(1&space;-&space;p_d)^{(1&space;-&space;x_d)}{\color{white}&space;}">
</p>

ou $d$ représente une dimension, $p_d \in \[0,1\]$ la probabilité que cette dimension prenne la valeur $1$ , $y \in \{classe\}$ la classe, et $x_d \in \{0,1\}$ la valeur possible.

Pour construire notre classifieur, nous cherchons à choisir la probabilité maximum d'une classe par rapport a l'image et a des paramètres que l'on va apprendre. on écrit cela ainsi :
<p align="center">
    <img src="https://latex.codecogs.com/png.image?\large&space;\dpi{110}\bg{black}y_{pred}&space;=&space;argmax_y&space;P(y&space;|&space;x_i&space;,&space;\theta){\color{white}&space;}">
</p>

On peut exprimer cette quantité de cette manière (formule expliquée en détail dans les notebooks):
<p align="center">
    <img src="https://latex.codecogs.com/png.image?\large&space;\dpi{110}\bg{black}f_{\theta}(x_i)&space;=&space;argmax_y&space;&space;\prod_{d}^{D}&space;p_{k,d}^{x_{i,d}}*(1&space;-&space;p_{k,d})^{(1&space;-&space;x_{i,d})}&space;*&space;\frac{1}{N}\sum_{i}^{N}&space;classe(x_i,y_j){\color{white}&space;}">
</p>

Les $p_{k,d,*}$ se calculent ainsi :
<p align="center">
  <img src="https://latex.codecogs.com/png.image?\large&space;\dpi{110}\bg{black}&space;p_{k,d,*}&space;&space;=&space;\frac{1}{|y|}\sum_{i}^{|y|}&space;x_{i,d}{\color{white}&space;}">
</p>

**Les calculs sont expliqués en détails dans les notebooks**

## Conclusion

A travers ce projet nous avons appris à développer une pipeline de la récupération des données à la mise en oeuvre d'un algorithme de classification multi-classe en expérimentant plusieurs types de pre processing. Nous avons obtenu des résultats corrects avec un meilleur compromis Nombre de paramètre / Performance pour la modélisation en patch de nos images.

## Reproduction des résultats

Pour reproduire les résultats de ce projet, il vous faudra clone le projet
``git clone https://github.com/jdufou1/L3_IAS_fruit_recognition_project.git``

Ensuite se rendre sur le dépot Kaggle des données et le télécharger à l'adresse : https://www.kaggle.com/datasets/moltean/fruits

Puis placez le dossier training à la racine du projet.
Enfin les notebooks se trouve dans ``./implementation``






