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

Une étape de pré-calcul des agrégations a été ajoutée avec le script :

`src/precompute_features.py`

Les agrégations les plus coûteuses sont désormais calculées une seule fois puis sauvegardées dans un fichier :

`data/processed/precomputed_aggs.joblib`

Le fichier `predict.py` a été modifié afin de :
- charger les agrégations pré-calculées ;
- réutiliser ces agrégations pendant l'inférence ;
- conserver le recalcul des features utilisateur dans `build_features()`.

Cette approche permet de :
- réduire fortement le coût du preprocessing ;
- conserver la cohérence métier des prédictions ;
- séparer le preprocessing offline de l'inférence online.

---

## Résultats après optimisation

| Métrique | Baseline | Optimized |
|---|---|---|
| Mean latency | 3.01 sec | 0.108 sec |
| Throughput | 0.33 req/sec | 9.28 req/sec |
| Memory usage | 2106 MB | 585 MB |
| Error rate | 0 % | 0 % |

---

## Justification de la configuration finale

Le modèle final conserve une architecture Scikit-learn chargée avec Joblib afin de garantir :
- une compatibilité simple avec l'environnement de déploiement ;
- une intégration stable dans l'API Gradio ;
- une maintenance légère.

Les agrégations les plus coûteuses sont pré-calculées puis stockées dans un fichier Joblib afin de réduire le coût du preprocessing pendant l'inférence.

Le profiling avec cProfile a montré que le principal bottleneck provenait des agrégations pandas réalisées dans `build_features()` et non de l'inférence du modèle lui-même.

L'utilisation d'ONNX Runtime a été envisagée mais n'a pas été retenue dans la configuration finale, car l'optimisation du preprocessing apportait déjà un gain significatif dans notre cas d'usage.

Le projet est exécuté sur CPU, ce qui reste suffisant pour ce modèle tabulaire après optimisation.

---

## Conclusion

Le pré-calcul des agrégations a permis de réduire fortement la latence d'inférence et d'améliorer significativement le throughput du système.

Cette optimisation rend le modèle beaucoup plus adapté à un usage temps réel tout en conservant la cohérence métier du pipeline de prédiction.