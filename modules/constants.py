from __future__ import annotations

"""Constantes partagées par tous les modules de classification."""

DIAMETRES_MM: dict[str, float] = {
    "1c": 16.25,
    "2c": 18.75,
    "5c": 21.25,
    "10c": 19.75,
    "20c": 22.25,
    "50c": 24.25,
    "1e": 23.25,
    "2e": 25.75,
}

VALEURS_CENTIMES: dict[str, int] = {
    "1c": 1,
    "2c": 2,
    "5c": 5,
    "10c": 10,
    "20c": 20,
    "50c": 50,
    "1e": 100,
    "2e": 200,
}

GROUPES: dict[str, list[str]] = {
    "cuivre": ["1c", "2c", "5c"],
    "or": ["10c", "20c", "50c"],
    "bimetallic": ["1e", "2e"],
}


def valeur_totale(valuations) -> tuple[int, str]:
    """Calcule la valeur totale en centimes et retourne un libellé lisible.

    Accepte toute liste d'objets ayant un attribut valeur_centimes.
    """
    total = sum(v.valeur_centimes for v in valuations)
    euros, centimes = divmod(total, 100)
    if euros > 0 and centimes > 0:
        libelle = f"{euros}e{centimes:02d}"
    elif euros > 0:
        libelle = f"{euros}e"
    else:
        libelle = f"{total}c"
    return total, libelle
