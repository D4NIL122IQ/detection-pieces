

**Reconnaissance automatique de pièces d'euros par traitement d'images**

Université Paris Cité — Traitement des Images Numériques (Sylvain Lobry), 2026.

---

## Approche

Le projet se décompose en deux phases évaluées indépendamment :

**Phase 1 — Détection** : localiser les pièces (transformée de Hough)
**Phase 2 — Classification** : identifier le type de chaque pièce (Random Forest)

---

## Résultats

### Détection (Hough)

| Métrique | Validation | Test |
|----------|------------|------|
| Rappel | 95.0% | 78.0% |
| Précision | 86.5% | 83.0% |
| **F1** | **90.6%** | **80.4%** |

### Classification (Random Forest, cross-validation 5-fold)

| Métrique | Score |
|----------|-------|
| **Accuracy** | **62.8%** |
| Macro F1 | 61.0% |
| Meilleurs : 1€ (85% F1), 2€ (81% F1) | |

---

## Méthodologie

### Dataset
~200 images annotées (LabelMe), divisées en 3 (60/20/20) :
- **Train** : entraînement du classifieur
- **Validation** : optimisation des paramètres Hough
- **Test** : évaluation finale (jamais utilisé pendant le développement)

### Paramètres Hough optimisés sur validation

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| TAILLE_MAX | 800 | Résolution de travail |
| BLUR_MEDIAN | 15 | Élimine le grain du bois |
| BLUR_GAUSS | 11 | Lisse les gradients |
| PARAM1 | 80 | Seuil Canny |
| PARAM2 | 40 | Seuil accumulateur |
| DP | 1.2 | Résolution accumulateur |
| RAYON_MIN | 3% image | Rayon minimum |
| RAYON_MAX | 30% image | Rayon maximum |

Prétraitement : CLAHE sur luminance (LAB) pour normaliser le contraste.

### Features (12 dimensions)

H, S, V (couleur HSV normalisée), rayon relatif, rayon image, détection bimetal, delta_s, polarité, canal B, texture, ratio R/G, nombre de pièces.

### Classification

Random Forest (500 arbres, scikit-learn). Post-correction : si le RF prédit bimetal (1€/2€) mais aucune structure bimetal détectée, reclassification automatique.

---

## Structure

```
EuroVision/
├── main.py                  # Entraînement + prédiction
├── app.py                   # Pipeline complet
├── interface.py             # Interface graphique
├── split_dataset.py         # Division val/test
├── metrique.py              # Métriques de détection
├── eval_classification.py   # Évaluation classification seule
├── eval_crossval.py         # Cross-validation
├── split.json               # Répartition du dataset
├── modele_knn.pkl           # Modèle RF entraîné
├── modules/
│   ├── segmentation.py      # Détection Hough
│   ├── features.py          # Extraction de caractéristiques
│   ├── classification.py    # Random Forest
│   ├── labelme_parser.py    # Lecture annotations LabelMe
│   └── chargement.py        # Chargement images
└── dataset/
    ├── images/
    └── annotations/
```

---

## Utilisation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy Pillow

python app.py --image dataset/images/img_001.jpg --output outputs/img_001_detected.jpg
python app.py --evaluate
python interface.py
```

---

## Etat actuel du repo

Premiere version implementee :
- lecture des annotations LabelMe dans `dataset/BDD`
- detection des pieces par transformee de Hough dans `modules/segmentation.py`
- evaluation precision / rappel / F1 via `app.py --evaluate`
- visualisation des cercles detectes dans `outputs/`

Note importante : dans l'etat actuel du dataset, les noms de `dataset/images` et `dataset/BDD` ne correspondent pas directement. L'evaluation utilise donc l'image embarquee dans chaque JSON LabelMe quand aucun fichier image associe n'est retrouve.
