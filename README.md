================
  Introduction
================

Ce projet propose une implémentation du papier GesRec pour faire de la reconnaissance de gestes issus du langage
des signes.

----------
 Keywords
----------

GesRec (https://arxiv.org/pdf/1901.10323.pdf) :

Ce papier propose de faire de la reconnaissance de geste en temps réel dans une vidéo.
Le modèle est constitué de 2 parties :
    1 - le detector : classifier binaire, ce modèle détecte la présence d'un geste dans la vidéo. Il prend en entrée
        un pack de 8 frames et donne en sortie 1 si un geste est détecté, et O si aucun geste n'est détecté.
    2 - le classifier : classifier multiclass, ce modèle classifie le geste detecté par le detector. Il prend en entrée
        un pack de 16 frames, et donne en sortir l'index du geste detecté.



LSF10 (s3://aoc-innov-data-dwyh/dataset_LSF10/LSF10.zip) :

Ce dataset constitué par l'équipe DWYH contient 10 mots en LSF:

    videos/:
                Chaque mot est signé 12 fois par 4 sujets, soit 48 vidéos par mots, et 480 vidéos en tout.

    annotations.csv:
                Ce fichier contient une ligne pour chaque vidéo :
                        video_id : l'id de la vidéo, dans videos/id
                        label_id : id du label dans le fichier labels.csv
                        total_frames : nombre de frames dans la vidéo
                        begin : frame à laquelle débute réellement le geste
                        end : frame à laquelle termine réellement le geste
                        subset : train ou val

    annotations_detector_train/val.csv :
                Chaque ligne de ce fichier est un pack de 8 frames (indiqué par les colonnes begin, end et padding)
                issus des vidéos. La colonne is_gesture (0 ou 1) permet de connaitre la classe du pack de 8 frames.

    annotations_classifier_train/val.csv :
                Chaque ligne de ce fichier est un pack de 16 frames (indiqué par les colonnes begin, end et padding)
                issus des vidéos. La colonne label_id (de 0 à 9) permet de connaitre la classe du pack de 16 frames,
                c'est à dire le geste effectué.

    phrases/: Ce dossier contient différentes séquence de 10 gestes par chacun des sujets.

    annotations_phrases.csv : Ce fichier contient la séquence réelle des gestes effectués dans les vidéos.



Jester (s3://aoc-innov-data-dwyh/dataset_jester/jester_dataset.zip):

Ce dataset est public et contient 27 gestes utilisés pour commander un ordinateur (swipe left, zoom with fingers..etc)
Plus d'infos à l'adresse officielle : https://20bn.com/datasets/jester


==========================
  Architecture du projet
==========================

data_preparation/ :
    - Ce dossier n'est nécéssaire que si vous souhaitez créer vous même votre dataset.
    - Il contient tous les scripts nécéssaires à la préparation des données à partir de LSF10_raw.
    - Vous avez le choix soit d'executer les scripts, soit de télécharger les données déjà retravaillées sur S3 (LSF10).

training/:
    - Ce dossier contient les notebooks jupyter permettant d'entrainer le modèle.

models/:
    - Ce dossier contient les modèles .h5 pré-entrainés

gesture_recognition/:
    - executer gesrec.py permet de tester directement les modèles à l'aide d'une vidéo ou de la webcam.


==========================
      Prise en main
==========================

---------------------------
  Setup de l'environnement
---------------------------

conda env update --file environment.yml
conda activate env-gesrec

----------------------------
 Execution de la demo seule
----------------------------

Si l'entrainement ne vous interresse pas et vous souhaitez seulement executer la demo, executez le script :
gesture_recognition/main.py
Ce script permet d'ouvrir la webcam ou une vidéo et d'analyser les gestes effectués dans la vidéo.
Par defaut, la demo ouvre toutes les vidéos de test (dossier phrases/) de LSF10 une par une.
Appuyer sur q pour fermer la fenêtre vidéo en cours.

Le texte le plus haut indique le résultat du Detector
Le texte juste en dessous donne le résultat du Classifier après application de la formule du OneTimeActivation
(Cette méthode est décrite dans le papier Gesrec)
La console permet de voir les résultats du Classifier sans appliquer la formule du OneTimeActivation

N'hésitez pas à modifier la fonction main pour ouvrir les vidéos de votre choix.
Pour ouvrir la webcam au lieu d'une vidéo il suffit de donner 'webcam' en argument à la fonction
real_time_video_analysis.

----------------------------
       Entrainement
----------------------------

La pipeline d'entrainement est la suivante:
    1. téléchargement des données
    2. préparation des données
    3. entrainement du Detector sur LSF10
    4. entrainement du Classifier sur Jester puis LSF10 via transfer learning (Jester -> LSF10)
    5. execution

1. téléchargement des données

Vous avez la possibilité de téléchager LSF10 sous forme de données brutes ou déjà préparées :

- LFS10_raw :   s3://aoc-innov-data-dwyh/dataset_LSF10/LSF10_raw.zip
- LSF10 :       s3://aoc-innov-data-dwyh/dataset_LSF10/LSF10.zip

Le zip Jester contient une version du dataset légèrement modifiée par rapport à la version téléchargeable sur le site
officiel :

- JESTER :      s3://aoc-innov-data-dwyh/dataset_jester/jester_dataset.zip

2. préparation des données

Les annotations de Jester et LSF10 sont fournis dans les zip mais il est possible de les regénérer en utilisant les
scripts process_jester.py et process_lsf10.py

3. Entrainement du detector

Utilisez votre gestionnaire de notebooks préféré (jupyter notebook, jupyter lab, ...)
Ouvrez detector_resnet.ipynb et executez toutes les cellules.

4. Entrainement du Classifier

Ouvrez detector_resnet.ipynb et executez toutes les cellules.

5. Execution :

Executez gesture_recognition/main.py en changeant la valeur de la variable MODELS_FOLDER en haut du script pour
'..\models\' au lieu de '..\models\best'

Vous avez également la possibilité de changer le modèle utilisé (DATASET='JESTER', ou DATASET='LSF10').
Pour tester en temps réel avec votre webcam, il vaut mieux utiliser JESTER, mais attention les gestes reconnus ne sont
pas les mêmes.


---------------

Merci d'avoir lu, si vous avez des questions ou des suggestions à faire, vous pouvez me contacter à l'adresse :
matthieud.75@gmail.com