# Projet Machine Learning : reconnaissance de fruits en fonction d'une image
Projet d'introduction à l'apprentissage statistique de la troisième année de Licence informatique de l'université Paris-Saclay\
Projet de reconnaissance de fruit en fonction d'une image implémenté avec M. Oscar Bruneau

# pipeline du projet :
![alt text](https://github.com/jdufou1/fruit_recognition/blob/main/img/pipeline.png)

Nous avons tout d'abord récupéré un DataSet d'image de 64x64 pixels sur le site de Kaggle.
Ensuite à partir de ces données, nous avons utilisé trois types de pre-processing :

<ul>
  <li>Traduction des images vers leur densité avec un histogramme de nuance de gris</li>
  <li>Divisions des images en patch avec clusturisation et application d'un algorithme KMeans pour associer a chaque image une classe</li>
  <li>Binarisation des pixels noir ou blanc</li>
</ul>

Ensuite à partir de ces nouvelles données transformés, nous avons appliqué un algorithme bayesien naif qui s'appuie sur la vraissemblance des données basé sur une loi binomiale.


# Performance et compléxité :
![alt text](https://github.com/jdufou1/fruit_recognition/blob/main/img/resultat.PNG)
