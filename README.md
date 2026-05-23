# Détection et classification de pièces d'euros

Projet de vision par ordinateur pour détecter automatiquement des pièces d'euros dans des images et en identifier la valeur, permettant le calcul du montant total présent dans une scène.

---

## Sommaire

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Pipeline global](#2-pipeline-global)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Dataset](#4-dataset)
5. [Étape 1 — Détection des pièces (segmentation)](#5-étape-1--détection-des-pièces-segmentation)
6. [Étape 2 — Classification de la valeur (determination)](#6-étape-2--classification-de-la-valeur-determination)
7. [Évaluation des performances](#7-évaluation-des-performances)
8. [Interface graphique](#8-interface-graphique)
9. [Limites et difficultés](#9-limites-et-difficultés)
10. [Perspectives](#10-perspectives)
11. [Installation et utilisation](#11-installation-et-utilisation)

---

## 1. Contexte et objectifs

### Problème traité

Ce projet répond à un problème classique de vision par ordinateur : **reconnaître et compter automatiquement des pièces de monnaie** dans une photographie.

L'application pratique est simple — pointer une caméra vers un ensemble de pièces d'euros et obtenir automatiquement :
- le nombre de pièces présentes
- la valeur de chacune (1¢, 2¢, 5¢, 10¢, 20¢, 50¢, 1€, 2€)
- le montant total

### Contraintes

- Pas d'apprentissage supervisé profond (pas de CNN) : le projet repose exclusivement sur des algorithmes classiques de traitement d'image (OpenCV).
- Les images sont prises dans des conditions variées : fond différent, éclairage non contrôlé, pièces partiellement superposées ou inclinées.
- La solution doit fonctionner sans calibration préalable (pas de pièce de référence dans l'image).

### Objectifs de performance

L'objectif initial était d'atteindre **70 % de précision** sur la détection. La version actuelle dépasse largement cet objectif :

| Tâche | Métrique | Train (199 img) | Test (119 img) |
|---|---|---|---|
| **Détection** | F1 | **82.64 %** | **83.21 %** |
| **Détection** | Précision | 82.29 % | 84.86 % |
| **Détection** | Rappel | 83.00 % | 81.62 % |
| **Classification valeur (8 classes)** | F1 micro | 36.2 % | **37.2 %** |
| **Classification valeur** | F1 macro | 36.1 % | **37.7 %** |

---

## 2. Pipeline global

```
Image d'entrée
      │
      ▼
┌──────────────────────────┐
│  Prétraitement            │  Redimensionnement, CLAHE adaptatif,
│  (segmentation.py)        │  flou médian + gaussien
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│  Détection Hough          │  HoughCircles (2 passes : principale + fallback)
│  (segmentation.py)        │  + détection gros plan + dédoublonnage géométrique
└──────────────────────────┘
      │
      ▼  Liste de cercles candidats
      │
      ▼
┌──────────────────────────┐
│  Validation               │  4 scores : edge, métallicité, couverture,
│  (validator.py)           │  circularité → vote 2/4
│                           │  + NMS par IoU (suppression doublons)
└──────────────────────────┘
      │
      ▼  Liste de cercles validés
      │
      ▼
┌──────────────────────────┐
│  Classification combinée  │  3 voteurs en parallèle :
│  (classification4.py)     │   - HSV + taille (classification.py)
│                           │   - Filtres RGB (classification3.py)
│                           │   - Profil radial + K-means (bimétal)
│                           │  → vote pondéré sur le groupe couleur
│                           │  → cohérence inter-pièces (ratios)
└──────────────────────────┘
      │
      ▼
Résultat : liste (valeur, confiance) + total en centimes
```

---

## 3. Architecture du projet

```
detection-pieces/
├── app.py                        # Point d'entrée CLI (détection)
├── eval_valeurs.py               # Évaluation de la classification des valeurs
├── interface.py                  # Application graphique (Tkinter)
├── main.py                       # Wrapper simplifié
├── metrique.py                   # Métriques de détection
├── metriqueVT.py                 # Métriques de classification par classe
├── optimize.py                   # Optimisation des paramètres
├── modules/
│   ├── __init__.py
│   ├── segmentation.py           # Détection des cercles (Hough)
│   ├── validator.py              # Validation + NMS par IoU
│   ├── classification.py         # Classification HSV (couleur + taille)
│   ├── classification2.py        # Variante HLS
│   ├── classification3.py        # Classification par filtres RGB
│   ├── classification4.py        # Vote combiné HSV + Filtres + bimétal
│   ├── constants.py              # Constantes partagées (diamètres, groupes)
│   ├── labelme_parser.py         # Lecture des annotations LabelMe
│   └── chargement.py             # Chargement image / annotation
└── dataset/
    ├── images/                   # 200 images + dataset test
    │   └── test/                 # 119 images de test
    └── BDD/                      # 199 annotations LabelMe (JSON)
        └── test/                 # 119 annotations test
```

### Rôle de chaque fichier

| Fichier | Rôle |
|---|---|
| `segmentation.py` | Pipeline de détection : prétraitement adaptatif, Hough avec fallback, détection gros plan, dédoublonnage |
| `validator.py` | Filtre les faux positifs via 4 scores (edge, métallicité, couverture, circularité) puis applique un NMS par IoU |
| `classification.py` | Classification par HSV : balance des blancs Gray-World + CLAHE + assignation groupe + ratios de taille |
| `classification2.py` | Variante en HLS (sigmoïde sur la teinte, frontière cuivre/or) |
| `classification3.py` | Classification par ratios RGB normalisés (R/(R+G+B), etc.) |
| `classification4.py` | **Méthode principale** : vote pondéré entre HSV + Filtres + score bimétal (profil radial + K-means k=2) + post-traitement de cohérence inter-pièces |
| `constants.py` | Diamètres officiels, dénominations, groupes couleur |
| `labelme_parser.py` | Parse les fichiers JSON LabelMe |
| `chargement.py` | Associe image à son annotation correspondante |
| `metrique.py` / `metriqueVT.py` | Calcul des métriques (détection / classification) |
| `app.py` | CLI détection : mode image unique ou évaluation complète |
| `eval_valeurs.py` | CLI évaluation des classifications de valeurs |
| `interface.py` | GUI Tkinter avec résultats superposés à l'image |
| `optimize.py` | Recherche des meilleurs paramètres par grid search |

---

## 4. Dataset

### Composition

- **Train** : 200 images JPEG (`img_001` à `img_200`) + 199 annotations LabelMe
- **Test** : 119 images additionnelles dans `dataset/images/test/` avec annotations dans `dataset/BDD/test/`
- Images prises dans des conditions variées : différents fonds, éclairages, angles et densités de pièces

### Format d'annotation (LabelMe)

Chaque annotation JSON décrit un ou plusieurs cercles. Pour chaque pièce :

```json
{
  "shapes": [
    {
      "label": "1cent",
      "shape_type": "circle",
      "points": [
        [cx, cy],     // centre du cercle
        [ex, ey]      // point sur le bord (permet de déduire le rayon)
      ]
    }
  ],
  "imagePath": "../images/img_XXX.jpg",
  "imageWidth": 1920,
  "imageHeight": 1440
}
```

Les labels utilisés sont : `1cent`, `2cent`, `5cent`, `10cent`, `20cent`, `50cent`, `1euro`, `2euro`.

### Réalignement des annotations

Les images ont été renommées après l'annotation initiale. Un script (`rename_bdd_annotations.py`) a été développé pour réaligner les annotations avec les nouvelles images en utilisant une **empreinte visuelle** : chaque image est réduite à 32×32 pixels en niveaux de gris et comparée au contenu embarqué dans le JSON (`imageData`).

---

## 5. Étape 1 — Détection des pièces (segmentation)

### Pourquoi la transformée de Hough ?

La transformée de Hough circulaire (`cv2.HoughCircles`) est adaptée à ce problème car :
- Les pièces sont des **formes quasi-parfaitement circulaires**
- L'algorithme est robuste à l'occlusion partielle (il ne nécessite pas de détecter l'intégralité du bord)
- Il ne nécessite aucun apprentissage

### Pipeline de prétraitement

Avant d'appeler Hough, l'image subit plusieurs étapes :

```
Image originale
    │
    ▼  Redimensionnement (max 800px)
    │       → normalise les paramètres Hough indépendamment de la résolution
    ▼  Conversion RGB → LAB
    │       → le canal L est traité indépendamment de la couleur
    ▼  CLAHE sur canal L (Contrast Limited Adaptive Histogram Equalization)
    │       → améliore le contraste local, révèle les bords de pièces peu contrastées
    │       → clipLimit=3.0, tileGridSize=8×8
    ▼  Flou médian (kernel 15)
    │       → réduit le bruit de texture (bois, tissu, fond texturé)
    ▼  Flou gaussien (kernel 11)
    │       → lisse pour éviter de faux bords dans Hough
    ▼  HoughCircles (gradient)
```

**Pourquoi CLAHE ?** L'égalisation classique de l'histogramme (global) sur-amplifie parfois le contraste dans les zones déjà bien éclairées. CLAHE opère sur des tuiles locales avec une limite de contraste, ce qui est plus adapté aux images avec un éclairage hétérogène.

### Paramètres de Hough

| Paramètre | Valeur | Signification |
|---|---|---|
| `dp` | 1.2 | Résolution de l'accumulateur (1 = même résolution que l'image) |
| `param1` | 80 | Seuil haut du détecteur de contours Canny interne |
| `param2` | 40 | Seuil de l'accumulateur (plus il est bas, plus on détecte) |
| `minDist` | 8 % de la taille image | Distance minimale entre deux centres |
| `minRadius` | 3 % de la taille image | Rayon minimum accepté |
| `maxRadius` | 30 % de la taille image | Rayon maximum accepté |

Les seuils sont **relatifs à la taille de l'image** (en pourcentage) pour être indépendants de la résolution.

### Dédoublonnage

Après Hough, des doublons peuvent apparaître (deux cercles quasi-identiques détectés pour la même pièce). Un filtre géométrique les supprime :

- **Cercles trop proches** : si la distance entre deux centres est inférieure à 60 % du rayon, l'un est un doublon
- **Cercles trop similaires** : si la différence de rayon est inférieure à max(4px, 35 % du rayon)
- **Cercles emboîtés** : un cercle dont le centre est très proche d'un plus grand est rejeté (anneau intérieur de détection)
- **Rayons aberrants** : si plusieurs pièces sont détectées et qu'un cercle a un rayon anormalement grand par rapport aux autres, il est supprimé

### Passe de secours (fallback)

Si aucun cercle n'est détecté en première passe, une seconde passe est lancée avec des paramètres plus permissifs :

| Paramètre | Valeur principale | Valeur fallback |
|---|---|---|
| `param2` | 40 | 28 |
| `minRadius` | 3 % | 2 % |
| `maxRadius` | 30 % | 35 % |

Cela permet de traiter les images où les pièces sont très petites, très peu contrastées ou prises sous un angle inhabituel.

### Détection gros plan

Quand de nombreux petits cercles sont détectés mais concentrés dans une zone restreinte sans cercle dominant (typique d'une photo en gros plan d'une seule pièce où les motifs internes génèrent des faux cercles), un post-traitement les fusionne en **un seul cercle englobant**.

### Validation et NMS

Après Hough, chaque cercle est validé par 4 scores indépendants (`modules/validator.py`) :

| Score | Description |
|---|---|
| `edge_score` | Force du gradient Sobel le long du périmètre (vrai bord de pièce → score élevé) |
| `metallic_score` | Cohérence de l'intérieur avec un disque métallique (luminosité modérée, saturation faible) |
| `coverage_score` | Proportion du disque effectivement dans l'image (rejette les pièces tronquées) |
| `circularity_score` | Circularité 4π·aire/périmètre² du contour détecté par Canny dans la ROI |

Un cercle est rejeté si au moins 2 des 4 critères échouent (**vote 2/4**).

Un **NMS par IoU** est ensuite appliqué : pour chaque paire de cercles dont l'IoU dépasse 2 %, seul celui qui a le meilleur score composite est conservé. Cette étape supprime les doublons résiduels que le dédoublonnage géométrique manque (gain de ~+5 points de F1).

---

## 6. Étape 2 — Classification de la valeur

### Principe général

Une fois les cercles détectés et validés, chaque pièce doit être identifiée parmi 8 valeurs. Le module `classification4.py` orchestre une **classification par vote pondéré** combinant plusieurs classifieurs indépendants.

### Les trois groupes de couleur

| Groupe | Pièces | Matériau | Teinte HSV |
|---|---|---|---|
| **Cuivre** | 1¢, 2¢, 5¢ | Acier recouvert de cuivre | Rouge-brun, H ≈ 11-14 |
| **Or nordique** | 10¢, 20¢, 50¢ | Alliage "Nordic Gold" | Doré, H ≈ 20-26 |
| **Bimétal** | 1€, 2€ | Deux alliages (centre + couronne) | Contraste centre/bord |

### Diamètres de référence

| Valeur | Diamètre | Groupe |
|---|---|---|
| 1¢ | 16.25 mm | Cuivre |
| 2¢ | 18.75 mm | Cuivre |
| 5¢ | 21.25 mm | Cuivre |
| 10¢ | 19.75 mm | Or nordique |
| 20¢ | 22.25 mm | Or nordique |
| 50¢ | 24.25 mm | Or nordique |
| 1€ | 23.25 mm | Bimétal |
| 2€ | 25.75 mm | Bimétal |

### Architecture : 3 voteurs

**Voteur 1 — HSV + taille** (`classification.py`)
1. Balance des blancs Gray-World pour corriger la dominante couleur
2. CLAHE sur le canal L de LAB (égalisation locale)
3. Conversion HSV, scores par groupe :
   - Cuivre/Or : score sigmoïde autour de H=17 + gaussien sur la teinte cible
   - Bimétal : score sur la proportion de pixels à faible saturation
4. Sélection de la valeur par minimisation de l'erreur sur les ratios de diamètre (test exhaustif pour N ≤ 3 pièces)

**Voteur 2 — Filtres RGB** (`classification3.py`)
- Ratios RGB normalisés invariants à l'éclairage : R/(R+G+B), G/(R+G+B), B/(R+G+B)
- Comparaison aux signatures de référence de chaque groupe
- Bénéficie d'un boost ×1.3 lors de la sélection de valeur dans le groupe cuivre

**Voteur 3 — Score bimétal centre/bord**
- **Profil radial** : trace la saturation HSV sur 20 anneaux concentriques de 0 à r, puis cherche le **saut maximal** entre deux moitiés du profil → un bimétal a une discontinuité abrupte ; un monométal a un profil lisse
- **K-means k=2** sur les canaux a,b de LAB (chrominance, ignore luminance) : un bimétal présente deux clusters bien séparés
- Fusion : `score = 0.8 × radial + 0.2 × kmeans`
- Détermine aussi 1€ vs 2€ selon quelle zone est saturée

### Vote sur le groupe couleur

Chaque voteur émet un vote pondéré par sa confiance :

```python
votes["cuivre"]     += conf_hsv         si HSV vote cuivre
votes["or"]         += conf_filtres     si Filtres vote or
votes["bimetallic"] += score_bimetal    si score_bimetal > 0.35
votes["bimetallic"] *= 0.5              si score_bimetal < 0.15  (pénalité monométal)
```

Le groupe avec le plus de votes est élu.

### Sélection de la valeur

Parmi les classifieurs alignés avec le groupe élu, le candidat avec la meilleure confiance gagne. Pour le groupe bimétal élu avec `score_bimetal > 0.35`, la suggestion 1€/2€ du voteur 3 est ajoutée comme candidat.

### Post-traitement : cohérence inter-pièces

Après classification individuelle, pour chaque groupe contenant ≥ 2 pièces dans l'image :

1. Trier les pièces par rayon croissant
2. Énumérer toutes les assignations possibles de dénominations (monotones, avec répétition)
3. Pour chaque assignation, calculer l'erreur = Σ |ratio_observé − ratio_théorique| sur les paires consécutives
4. Garder l'assignation à erreur minimale si erreur < 0.30 et confiance moyenne < 0.75

Cela corrige les classifications individuelles incohérentes en exploitant les contraintes physiques (les diamètres réels imposent des ratios précis entre pièces).

---

## 7. Évaluation des performances

### Détection des cercles

| Métrique | Train (199 img) | Test (119 img) |
|---|---|---|
| Précision | 82.29 % | **84.86 %** |
| Rappel | 83.00 % | 81.62 % |
| **F1** | **82.64 %** | **83.21 %** |
| TP / FP / FN | 576 / 124 / 118 | 342 / 61 / 77 |

### Classification de la valeur (8 classes)

| Métrique | Train | Test |
|---|---|---|
| **F1 micro** | 36.2 % | **37.2 %** |
| **F1 macro** | 36.1 % | **37.7 %** |
| Précision | 36.0 % | 38.0 % |
| Rappel | 36.3 % | 36.5 % |

**Détail par classe (test)** :

| Classe | TP | Précision | Rappel | F1 |
|---|---|---|---|---|
| 1¢ | 17 | 68.0 % | 37.8 % | **48.6 %** |
| 2¢ | 8 | 53.3 % | 32.0 % | 40.0 % |
| 5¢ | 9 | 32.1 % | 29.0 % | 30.5 % |
| 10¢ | 17 | 30.4 % | 25.0 % | 27.4 % |
| 20¢ | 26 | 44.8 % | 25.5 % | 32.5 % |
| 50¢ | 23 | 35.4 % | 44.2 % | 39.3 % |
| 1€ | 20 | 27.0 % | 45.5 % | 33.9 % |
| 2€ | 33 | 40.2 % | 63.5 % | **49.3 %** |

### Définitions

- **Précision** = TP / (TP + FP) : parmi les cercles détectés, combien correspondent vraiment à une pièce ?
- **Rappel** = TP / (TP + FN) : parmi les vraies pièces, combien ont été détectées ?
- **F1** = 2 × (précision × rappel) / (précision + rappel) : moyenne harmonique
- **F1 micro** vs **macro** : micro pondère par classe selon le nombre d'instances ; macro fait une moyenne simple sur les classes

### Critère de correspondance (détection)

Un cercle détecté est considéré comme **vrai positif** si :
- Distance entre le centre détecté et le centre annoté < max(10px, 60 % du rayon annoté)
- Différence de rayon < max(8px, 45 % du rayon annoté)

L'appariement est **greedy** : on associe en priorité le cercle détecté le plus proche de chaque annotation en minimisant l'erreur de rayon.

### Analyse

- **Détection** : la précision est passée de 72 % à 82 % grâce au NMS par IoU, sans perte notable de rappel.
- **Classification** : le score F1 a doublé par rapport à la version initiale (~18 % → 37 %), avec un gain particulièrement marqué sur les bimétaux (2€ à 49.3 %).
- **Robustesse** : les performances sur la base test sont équivalentes voire supérieures à la base train → pas de sur-apprentissage.

---

## 8. Interface graphique

L'interface (`interface.py`) est développée avec **Tkinter** (bibliothèque standard Python).

### Fonctionnalités

- **Chargement d'image** via explorateur de fichiers
- **Analyse automatique** : détection + classification au chargement
- **Affichage visuel** :
  - Cercles verts = pièces détectées par le système
  - Étiquette de valeur au-dessus de chaque cercle
- **Panneau de résultats** :
  - Résolution de l'image
  - Nombre de pièces détectées
  - **Montant total** calculé (ex : `3€ 45`)
  - Tableau détaillé : valeur, groupe de couleur, confiance, rayon en pixels

### Lancement

```bash
python interface.py
```

---

## 9. Limites et difficultés

### Difficultés de détection

| Problème | Cause | Impact |
|---|---|---|
| Fonds texturés (bois, tissu) | Bords parasites détectés comme cercles | Faux positifs |
| Pièces peu contrastées | CLAHE insuffisant sur certaines zones | Faux négatifs |
| Objets circulaires non monétaires | L'algorithme ne connaît pas le contexte | Faux positifs |
| Pièces superposées | Hough ne détecte pas les arcs incomplets | Faux négatifs |
| Images très sombres ou surexposées | Mauvaise réponse de Canny interne | Faux négatifs |

### Difficultés de classification

| Problème | Cause |
|---|---|
| 1 seule pièce par groupe présente | Impossible d'utiliser les ratios → dépend uniquement de la couleur |
| Éclairage coloré | Teinte HSV faussée → mauvais groupe assigné |
| Pièces usées ou sales | Couleur altérée |
| Pièces étrangères similaires | Non dans la base de référence |

### Absence de calibration

Sans pièce de référence dans l'image et sans information sur la distance focale de la caméra, il est impossible de mesurer les diamètres réels en millimètres. La classification par taille repose donc uniquement sur les **rapports de taille entre pièces** dans la même image.

---

## 10. Perspectives

### Améliorations possibles de la détection

- **Multi-scale Hough** : traiter l'image à plusieurs résolutions pour récupérer les ~17 % de pièces non détectées
- **Détecteur RANSAC sur contours** : alternative à Hough sur images très texturées
- **Deep learning** : remplacer Hough par un détecteur d'objets (YOLOv8) pour être plus robuste

### Améliorations possibles de la classification

- **Calibration automatique** : détecter une pièce connue dans l'image pour calculer l'échelle réelle
- **Descripteurs de texture** (LBP, GLCM) : exploiter la texture distinctive (étoiles de la couronne bimétal)
- **Balance des blancs locale** par pièce plutôt que globale
- **Apprentissage supervisé** : entraîner un CNN léger (MobileNet) sur le dataset → atteindrait probablement > 90 %

### Plafond classique atteint

Sans pièce de référence ni deep learning, le système est proche de son plafond théorique. Les limitations restantes sont structurelles :
- Ambiguïté géométrique entre 5¢/20¢ et 50¢/2€ (tailles trop proches)
- Images mono-pièce sans référence (impossible à calibrer)
- Pièces usées dont la couleur ne distingue plus cuivre/or

### Extensions fonctionnelles

- Export JSON des résultats
- Mode vidéo en temps réel
- Application mobile

---

## 11. Installation et utilisation

### Prérequis

Python 3.9+ requis.

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
pip install opencv-python numpy Pillow
```

### Utilisation

#### Tester une image

```bash
python app.py --image dataset/images/img_001.jpg
```

#### Sauvegarder le résultat annoté

```bash
python app.py --image dataset/images/img_001.jpg --output outputs/img_001_detected.jpg
```

#### Évaluer la détection sur tout le dataset

```bash
python app.py --evaluate
```

#### Évaluer la détection sur la base test

```bash
python app.py --evaluate --images-dir dataset/images/test --annotations-dir dataset/BDD/test
```

#### Évaluer la classification des valeurs

```bash
python eval_valeurs.py                                              # train
python eval_valeurs.py --images-dir dataset/images/test \
                      --annotations-dir dataset/BDD/test            # test
```

#### Évaluer sur N images avec sauvegarde des visualisations

```bash
python app.py --evaluate --limit 50 --vis-dir outputs/eval/
```

#### Lancer l'interface graphique

```bash
python interface.py
```

---

## Résumé technique

| Aspect | Choix technique |
|---|---|
| Langage | Python 3.9+ |
| Bibliothèque principale | OpenCV (`cv2`) |
| Détection | HoughCircles + fallback + détection gros plan |
| Validation | 4 scores (edge, métallicité, couverture, circularité) + NMS par IoU |
| Prétraitement | CLAHE adaptatif selon exposition + flou médian + gaussien |
| Classification | Vote pondéré : HSV + Filtres RGB + Profil radial bimétal + K-means |
| Post-traitement | Cohérence inter-pièces par ratios de diamètre |
| Annotation | Format LabelMe (JSON) |
| Interface | Tkinter |
| Dataset | 200 images train + 119 images test |
| **Performances détection (test)** | **P 84.86 %, R 81.62 %, F1 83.21 %** |
| **Performances classification (test)** | **F1 micro 37.2 %, F1 macro 37.7 %** |
