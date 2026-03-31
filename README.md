# Détection de pièces d'euros

Projet de vision par ordinateur pour détecter automatiquement des pièces d'euros dans des images.

La première étape du projet consiste à localiser les pièces avec une transformée de Hough.  
La classification des valeurs viendra dans une deuxième étape.

## Objectif

Le but est de construire un pipeline simple et robuste capable de :
- charger une image
- détecter les pièces présentes
- afficher les cercles détectés
- évaluer les performances sur un dataset annoté

## État actuel

La détection est déjà fonctionnelle et atteint actuellement sur le dataset annoté :
- précision : `70.33%`
- rappel : `84.73%`
- F1 : `76.86%`

Évaluation réalisée sur `199` annotations disponibles.

Le niveau visé au départ était d'environ `70%`, donc cette version dépasse déjà cet objectif malgré plusieurs images difficiles.

## Dataset

Le dataset est organisé ainsi :

```text
dataset/
├── images/
└── BDD/
```

- `dataset/images` contient les images renommées sous la forme `img_001`, `img_002`, etc.
- `dataset/BDD` contient les annotations LabelMe correspondantes

Les annotations ont été réalignées avec les images renommées.

État actuel :
- `200` images dans `dataset/images`
- `199` annotations dans `dataset/BDD`

Il manque donc encore une annotation.

## Organisation du projet

```text
.
├── app.py
├── interface.py
├── main.py
├── metrique.py
├── rename_bdd_annotations.py
├── dataset/
│   ├── images/
│   └── BDD/
└── modules/
    ├── __init__.py
    ├── chargement.py
    ├── labelme_parser.py
    └── segmentation.py
```

## Fichiers principaux

- `modules/segmentation.py`
  Détection des pièces avec `cv2.HoughCircles`

- `modules/labelme_parser.py`
  Lecture des annotations LabelMe

- `modules/chargement.py`
  Appariement image / annotation

- `metrique.py`
  Calcul des métriques de détection

- `app.py`
  Exécution en ligne de commande

- `interface.py`
  Interface graphique locale

- `rename_bdd_annotations.py`
  Script pour réaligner les annotations avec les images renommées

## Méthode de détection

La détection repose sur :
- redimensionnement avec taille max `800`
- amélioration locale du contraste avec CLAHE
- flou médian puis gaussien
- détection circulaire avec la transformée de Hough
- suppression légère de doublons
- seconde passe plus permissive uniquement si rien n'est détecté

Paramètres principaux :
- `DP = 1.2`
- `PARAM1 = 80`
- `PARAM2 = 40`
- rayon minimum relatif : `3%`
- rayon maximum relatif : `30%`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy Pillow
```

## Utilisation

### Tester une image

```bash
python app.py --image dataset/images/img_001.jpg
```

### Sauvegarder une image annotée

```bash
python app.py --image dataset/images/img_001.jpg --output outputs/img_001_detected.jpg
```

### Évaluer le dataset

```bash
python app.py --evaluate
```

### Lancer l'interface graphique

```bash
python interface.py
```

### Vérifier le renommage des annotations

```bash
python rename_bdd_annotations.py --dry-run
```

### Appliquer le renommage

```bash
python rename_bdd_annotations.py
```

## Limites actuelles

- certaines scènes complexes produisent encore des faux positifs
- les fonds bois ou tissus restent difficiles
- certaines pièces très peu contrastées sont encore ratées
- des objets circulaires non monétaires peuvent parfois être détectés

## Suite prévue

La prochaine étape est la classification des pièces :
- extraction de caractéristiques
- entraînement d'un classifieur
- ajout de la prédiction de valeur dans le pipeline complet
