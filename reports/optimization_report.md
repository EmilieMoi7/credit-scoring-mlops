# Analyse et optimisation des performances du modèle

## Baseline de performance

Un benchmark de 50 requêtes a été réalisé afin de mesurer les performances initiales du système.

### Résultats baseline

- Latence moyenne : 3.01 sec
- P95 : 3.31 sec
- Throughput : 0.33 req/sec
- CPU moyen : 24.49 %
- Mémoire moyenne : 2106 MB
- Taux d'erreur : 0 %

Les résultats montrent une latence relativement élevée pour un modèle de credit scoring.

---

## Profiling avec cProfile

Un profiling a été réalisé avec cProfile afin d'identifier les goulots d'étranglement.

### Résultats du profiling

Le profiling montre que la majorité du temps d'exécution est consommée par la fonction `build_features()` appelée depuis `prepare_input()`.

Les fonctions les plus coûteuses identifiées sont :

- `aggregate_installments`
- `aggregate_bureau`

Ces fonctions réalisent des agrégations pandas coûteuses (`groupby`, `aggregate`) sur plusieurs datasets volumineux.

### Conclusion du profiling

Le principal bottleneck identifié n'est pas le modèle de machine learning lui-même mais le preprocessing et les agrégations de features recalculées à chaque requête.

---

## Optimisation des performances

### Solution mise en place

Une étape de pré-calcul des features a été ajoutée avec le script :

`src/precompute_features.py`

Les features sont désormais calculées une seule fois puis sauvegardées dans un fichier parquet :

`data/processed/features_cache.parquet`

Le fichier `predict.py` a été modifié afin de :
- charger directement les features pré-calculées ;
- éviter les appels répétés à `build_features()` pendant l'inférence.

Cette approche permet de séparer :
- le preprocessing offline ;
- l'inférence online.

---

## Résultats après optimisation

| Métrique | Baseline | Optimized v1 |
|---|---|---|
| Mean latency | 3.01 sec | 0.006 sec |
| Throughput | 0.33 req/sec | 159 req/sec |
| Memory usage | 2106 MB | 357 MB |
| Error rate | 0 % | 0 % |

---

## Conclusion

L'utilisation d'un cache de features pré-calculées a permis de réduire fortement la latence d'inférence et d'améliorer significativement le throughput du système.

Cette optimisation rend le modèle beaucoup plus adapté à un usage temps réel tout en conservant les mêmes features et le même pipeline de machine learning.