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

### Résultats

Le profiling montre que le temps d'exécution est principalement consommé par la fonction `build_features()` appelée depuis `prepare_input()`.

Les fonctions les plus coûteuses sont :

- `aggregate_installments`
- `aggregate_bureau`

Ces fonctions réalisent des agrégations pandas coûteuses sur plusieurs datasets volumineux.

### Conclusion

Le principal bottleneck identifié n'est pas le modèle de machine learning lui-même mais le preprocessing et les agrégations de features recalculées à chaque requête.