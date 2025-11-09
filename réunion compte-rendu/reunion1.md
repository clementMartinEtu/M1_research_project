# Compte-rendu du projet de classification de fleurs

## Présentation

Nous avons vu de mon côté, quels étaient les objectifs à atteindre pour savoir s'il est possible de créer un classifier qui annote les fleurs pour une image fixe à haute résolution.

Nous avons discuté de l'article [Automatic flower detection and phenology monitoring using time-lapse cameras and deep learning](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.275), qui présente une méthode de détection de fleurs à partir d'un réseau de neurones et qui annote sur ces images l'espèce de la fleur.

## Objectifs

Des objectifs ont été relevés afin d'avancer sur le projet, tout d'abord quel est le type de modèle utilisé pour appliquer le deep learning et quelles sont les librairies nécessaires pour cela.

Une question qui se posait était de savoir s'il fallait un seul classifier pour détecter toutes les potentielles espèces de fleurs ou s'il fallait créer plusieurs classifiers et donc en dernier point, peut-on avoir une complexité raisonnable pour ce type de modèle.

Concernant le modèle de réseaux de neurones, on s'est demandé s'il fallait l'entraîner avant de l'utiliser ou alors s'il existe déjà un modèle entraîné à la tâche que l'on souhaite. Mais aussi, est-il possible d'améliorer l'entraînement s'il existe.

Je dois aussi essayer de trouver un dataset annoté de botanique et si jamais je ne trouve pas, essayer avec un autre type de dataset annoté comme des animaux pour vérifier que le modèle fonctionne toujours. C'est une idée intéressante pour possiblement utiliser ce modèle de détection dans d'autres projets dans d'autres domaines.

## Contribution de Kelian

Kelian a pris l'autre direction du projet qui est la potentielle classification à partir d'une vue aérienne donc une vue bien plus large mais de moins bonne qualité en utilisant un drone qui possède une caméra multispectrale. L'utilisation des images multispectrales est intéressante puisque la végétation a des "couleurs" différentes et beaucoup plus variées qu’en restant dans le domaine du visible et donc la reconnaissance de ces fleurs est plus facile.

Il a étudié une méthode de détection basée sur :
1. Application du PCA pour réduire la dimension de l'image satellite.
2. Utilisation de l'algorithme K-means pour effectuer un clustering.
3. Retour à la dimension initiale de l'image.  

Deux questions se sont posées, est-il possible d'implémenter un réseau de neurones pour faire la même chose et aussi est-ce que les infrastructures gênent la détection des fleurs ?

## Lien avec nos recherches

On peut faire un lien avec mes images fixes, en utilisant ce type de caméras multispectrales, est-ce qu'on pourrait obtenir une précision plus importante ? Mais ce modèle n'est peut-être pas possible puisque les images que l'on utilise comportent plus de données et la complexité pourrait ne plus être raisonnable pour une utilisation à grande échelle.
