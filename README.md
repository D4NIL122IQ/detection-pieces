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

L'objectif initial était d'atteindre **70 % de précision** sur la détection. La version actuelle dépasse cet objectif avec un **F1-score de 76.86 %**.

---

## 2. Pipeline global

```
Image d'entrée
      │
      ▼
┌─────────────────────┐
│  Prétraitement       │  Redimensionnement, CLAHE, flou médian, flou gaussien
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Détection           │  Transformée de Hough circulaire (HoughCircles)
│  (segmentation.py)   │  + dédoublonnage géométrique
└─────────────────────┘
      │
      ▼  Liste de cercles (x, y, rayon)
      │
      ▼
┌─────────────────────┐
│  Classification      │  Analyse couleur HSV + ratios de taille
│  (determination.py)  │  → groupe (cuivre / or nordique / bimétal)
│                      │  → valeur précise (1¢ … 2€)
└─────────────────────┘
      │
      ▼
Résultat : liste (valeur, confiance) + total en centimes
```

---

## 3. Architecture du projet

```
detection-pieces/
├── app.py                        # Point d'entrée CLI
├── interface.py                  # Application graphique (Tkinter)
├── main.py                       # Wrapper simplifié
├── metrique.py                   # Calcul des métriques de détection
├── rename_bdd_annotations.py     # Outil de gestion du dataset
├── modules/
│   ├── __init__.py
│   ├── segmentation.py           # Détection des cercles (Hough)
│   ├── determination.py          # Classification des valeurs
│   ├── labelme_parser.py         # Lecture des annotations LabelMe
│   └── chargement.py             # Chargement image / annotation
└── dataset/
    ├── images/                   # 200 images (img_001 … img_200)
    └── BDD/                      # 199 annotations LabelMe (JSON)
```

### Rôle de chaque fichier

| Fichier | Rôle |
|---|---|
| `segmentation.py` | Contient tout le pipeline de détection : prétraitement, appel Hough, filtrage des doublons |
| `determination.py` | Analyse couleur + taille de chaque cercle détecté pour déterminer la valeur de la pièce |
| `labelme_parser.py` | Parse les fichiers JSON au format LabelMe pour extraire les annotations de vérité terrain |
| `chargement.py` | Associe chaque image à son annotation correspondante |
| `metrique.py` | Calcule précision, rappel, F1 en comparant détections et annotations |
| `app.py` | CLI : mode image unique ou évaluation complète du dataset |
| `interface.py` | GUI Tkinter avec affichage des résultats superposés à l'image |
| `rename_bdd_annotations.py` | Réaligne les annotations avec les images renommées via empreinte visuelle |

---

## 4. Dataset

### Composition

- **200 images** au format JPEG, nommées `img_001` à `img_200`
- **199 annotations** au format LabelMe (JSON), une par image (une image sans annotation)
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

---

## 6. Étape 2 — Classification de la valeur (determination)

### Principe général

Une fois les cercles détectés, chaque pièce doit être identifiée parmi les 8 valeurs possibles. L'approche est basée sur deux caractéristiques physiques des pièces :

1. **La couleur** : les pièces d'euros sont fabriquées en trois types de matériaux distincts
2. **La taille** : chaque valeur a un diamètre réglementé

### Les trois groupes de couleur

| Groupe | Pièces | Matériau | Teinte HSV |
|---|---|---|---|
| **Cuivre** | 1¢, 2¢, 5¢ | Acier recouvert de cuivre | Rouge-brun, H ≈ 11-14 |
| **Or nordique** | 10¢, 20¢, 50¢ | Alliage "Nordic Gold" | Doré, H ≈ 20-26 |
| **Bimétal** | 1€, 2€ | Deux alliages (centre + couronne) | Faible saturation globale |

### Passe 1 : Assignation au groupe de couleur

Pour chaque pièce détectée :
1. La région circulaire est extraite et normalisée (64×64 pixels)
2. L'image est convertie en **HSV** (Hue, Saturation, Value)
3. Un **score de couleur** est calculé pour chaque groupe :

**Pour Cuivre et Or nordique :**
- Les pixels sont pondérés par leur saturation (pixels gris ignorés)
- Un score sigmoïde est calculé autour de H = 17 (frontière cuivre/or)
- Un score gaussien est ajouté centré sur la teinte cible du groupe
- Un seuil adaptatif sur V (25 % de la médiane locale) filtre les pixels trop sombres

**Pour Bimétal :**
- Score basé sur la **faible saturation** : les pièces bimétal ont une apparence globalement grisée (argent dominant)
- Calcul du pourcentage de pixels fortement saturés (signature cuivre/or absent)

Le groupe retenu est celui avec le score le plus élevé.

### Passe 2 : Sélection de la valeur dans le groupe

Une fois le groupe connu, on détermine la valeur précise parmi les 2 ou 3 candidats en utilisant les **ratios de diamètre**.

**Diamètres de référence réels (en mm) :**

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

L'algorithme trie les pièces détectées par rayon et les candidats par diamètre, puis cherche la combinaison qui **minimise l'erreur relative sur tous les ratios de diamètre** entre paires de pièces. Pour N ≤ 3 pièces, toutes les combinaisons sont testées.

### Cas particulier : 1€ vs 2€

Pour distinguer 1€ de 2€ lorsqu'une seule pièce bimétal est présente, l'algorithme analyse la distribution couleur **centre vs couronne** :

- **1€** : centre en argent (acier) + couronne en or nordique
- **2€** : centre en or nordique + couronne en argent (acier)

En plus de l'analyse couleur, un bonus de taille est appliqué (2€ > 1€ en diamètre).

### Score de confiance

Chaque prédiction est accompagnée d'un **score de confiance** (0 à 1) qui reflète la certitude de la classification. Ce score peut être utilisé pour filtrer les prédictions douteuses.

---

## 7. Évaluation des performances

### Métriques

| Métrique | Valeur |
|---|---|
| Précision | **70.33 %** |
| Rappel | **84.73 %** |
| F1-score | **76.86 %** |

Évaluation réalisée sur **199 images annotées**.

### Définitions

- **Précision** = TP / (TP + FP) : parmi les cercles détectés, combien correspondent vraiment à une pièce ?
- **Rappel** = TP / (TP + FN) : parmi les vraies pièces, combien ont été détectées ?
- **F1** = 2 × (précision × rappel) / (précision + rappel) : moyenne harmonique

### Critère de correspondance

Un cercle détecté est considéré comme **vrai positif** si :
- Distance entre le centre détecté et le centre annoté < max(10px, 60 % du rayon annoté)
- Différence de rayon < max(8px, 45 % du rayon annoté)

L'appariement est **greedy** : on associe en priorité le cercle détecté le plus proche de chaque annotation en minimisant l'erreur de rayon.

### Analyse

Le **rappel (84.73 %)** est nettement supérieur à la **précision (70.33 %)**, ce qui signifie que :
- Le système manque peu de pièces (bon rappel)
- Mais génère un certain nombre de faux positifs (objets circulaires non monétaires détectés)

Ce comportement est intentionnel : mieux vaut sur-détecter et filtrer ensuite que rater des pièces.

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

- **Filtrage par forme** : calculer le ratio d'aspect ou la circularité pour rejeter les faux positifs non circulaires
- **Deep learning** : remplacer Hough par un détecteur d'objets (YOLOv8) pour être plus robuste aux fonds complexes
- **Multi-scale detection** : traiter l'image à plusieurs résolutions pour mieux gérer les pièces très petites ou très grandes

### Améliorations possibles de la classification

- **Calibration automatique** : détecter une pièce connue dans l'image pour calculer l'échelle
- **Descripteurs de texture** : analyser la texture (face/pile) pour une classification plus fiable
- **Apprentissage supervisé** : entraîner un CNN sur un dataset de régions de pièces recadrées
- **Gestion des bimétaux** : améliorer l'analyse centre/couronne pour mieux distinguer 1€ et 2€

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

#### Évaluer sur tout le dataset

```bash
python app.py --evaluate
```

#### Évaluer sur N images avec sauvegarde des visualisations

```bash
python app.py --evaluate --limit 50 --vis-dir outputs/eval/
```

#### Lancer l'interface graphique

```bash
python interface.py
```

#### Vérifier le réalignement des annotations (sans modifier)

```bash
python rename_bdd_annotations.py --dry-run
```

#### Appliquer le réalignement

```bash
python rename_bdd_annotations.py
```

---

## Résumé technique

| Aspect | Choix technique |
|---|---|
| Langage | Python 3.9+ |
| Bibliothèque principale | OpenCV (`cv2`) |
| Détection | Transformée de Hough circulaire |
| Prétraitement | CLAHE + flou médian + flou gaussien |
| Classification | Analyse couleur HSV + ratios de diamètre |
| Annotation | Format LabelMe (JSON) |
| Interface | Tkinter |
| Dataset | 200 images, 199 annotations |
| Performances | Précision 70.33 %, Rappel 84.73 %, F1 76.86 % |
