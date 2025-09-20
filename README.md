# poc-aws-reco-ml100k

*(English below)*

## 🇫🇷 Présentation
Ce projet est un **POC minimaliste** visant à démontrer la mise en place d’un **système de recommandation** à partir du dataset [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/).  

L’objectif n’est pas de créer un modèle complexe mais de **montrer une chaîne complète** :  
- Extraction et préparation des données  
- Génération de recommandations personnalisées  
- Export des résultats  
- Déploiement et exécution sur **AWS EC2**  

👉 Résultat : pour un utilisateur donné, on obtient une liste de films recommandés (top-k) avec un score de pertinence.

---

## Stack technique
- **Python 3** (pandas, numpy)  
- **MovieLens 100k** (dataset public de recommandations cinéma)  
- **AWS EC2** pour exécuter le projet dans le cloud  
- **S3** pour stocker les données (optionnel)  

---

## Utilisation locale
1. Cloner le repo :  
   ```bash
   git clone https://github.com/<ton-user>/poc-aws-reco-ml100k.git
   cd poc-aws-reco-ml100k
## Installer la dépendance
pip install -r requirements.txt

## Lancer une recommandation
python poc_reco_kaggle_ml100k.py --user 42 --topk 10 --alpha 0.5

## Résultat attendu
- Fargo (1996) — score 0.764
- The Godfather (1972) — score 0.735
- Pulp Fiction (1994) — score 0.727
...
CSV écrit: outputs/ml100k_recos_user_42.csv

## création clé SSH
## Lancement d'une instance ubuntu (pour moi)
aws ec2 run-instances \
  --image-id <AMI_ID> \
  --count 1 \
  --instance-type t3.micro \
  --key-name <KEY_NAME> \
  --security-group-ids <SG_ID>
  
## connexion
ssh -i ~/.ssh/<KEY_NAME> ubuntu@<PUBLIC_IP>

## déploiement

Montrer la logique d’un système de recommandation simple.

Démontrer l’intégration Data Science + Cloud AWS.

Servir de base à des projets plus avancés (API, bases de données, scalabilité).

