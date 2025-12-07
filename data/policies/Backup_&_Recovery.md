# Politique de Backup & Recovery 

## 1. Objet
La présente politique définit les règles, procédures et obligations relatives à la sauvegarde des données, à la restauration des systèmes et à la continuité des opérations. Elle vise à assurer la disponibilité, l’intégrité et la résilience des systèmes informatiques de l’entreprise afin de réduire les risques de perte de données, d’interruption de service ou d’impact opérationnel majeur.

## 2. Portée
Cette politique s’applique à :
- Tous les systèmes d’information (serveurs, bases de données, applications internes, applications cloud).
- Tous les environnements de production, pré-production, staging et QA.
- Les données critiques, sensibles ou opérationnelles.
- Les équipes techniques, administrateurs systèmes, ingénieurs cloud et responsables applicatifs.

Aucune exception n’est tolérée sans approbation écrite de la direction IT.

## 3. Objectifs de la Sauvegarde
Les objectifs principaux sont :
- Prévenir toute perte définitive de données.
- Permettre une restauration rapide en cas d’incident.
- Réduire les interruptions de service.
- Maintenir la conformité légale, réglementaire et contractuelle.
- Assurer un niveau de continuité acceptable pour les utilisateurs internes et externes.

## 4. Types de Données à Sauvegarder
Les catégories suivantes doivent obligatoirement être sauvegardées :
- Bases de données opérationnelles
- Fichiers de configuration système
- Environnements applicatifs critiques
- Fichiers utilisateurs et documents internes
- Journaux d’activité critiques (logs systèmes)
- Machines virtuelles ou containers critiques

Les données temporaires ou facilement régénérables ne doivent pas être sauvegardées.

## 5. Fréquence des Sauvegardes
Les fréquences standards sont définies comme suit :

### 5.1 Systèmes Critiques
- Sauvegarde complète quotidienne.
- Sauvegardes incrémentales toutes les 4 heures.
- Rétention minimale : 14 jours.

### 5.2 Systèmes Non Critiques
- Sauvegarde complète hebdomadaire.
- Incrémentales quotidiennes.
- Rétention minimale : 7 jours.

### 5.3 Données Sensibles
- Sauvegarde complète quotidienne.
- Chiffrement obligatoire.
- Rétention : 30 jours.

### 5.4 Données Long Terme
- Archivage trimestriel.
- Rétention minimale : 3 ans.

Les fréquences doivent être documentées dans le plan de sauvegarde.

## 6. Méthodes de Sauvegarde
Les méthodes suivantes sont autorisées :
- Sauvegarde complète (Full Backup)
- Sauvegarde incrémentale
- Sauvegarde différentielle
- Snapshot système (VM, containers, volumes)

Les sauvegardes manuelles non tracées sont interdites.

## 7. Localisation des Sauvegardes
Les sauvegardes doivent être stockées dans :
- Un datastore primaire sécurisé
- Un datastore secondaire hors site
- Une solution cloud validée par la sécurité

Les sauvegardes ne doivent jamais être stockées sur :
- PC personnels
- Clés USB
- Emails internes ou externes

## 8. Chiffrement des Sauvegardes
Toutes les sauvegardes doivent être chiffrées selon les standards suivants :
- AES-256 pour les données au repos
- TLS 1.2+ pour les données en transit
- Mots de passe et clés stockés dans un coffre-fort numérique (Vault)

Toute sauvegarde non chiffrée constitue une violation de sécurité critique.

## 9. Restauration des Données
Les restaurations doivent respecter :
- Une demande formelle ouverte via un ticket
- Une validation par le manager pour les données sensibles
- Une documentation de chaque opération

Les restaurations doivent être testées régulièrement pour garantir leur efficacité.

## 10. Tests de Restauration
Les tests doivent être réalisés :
- Tous les 6 mois pour les systèmes critiques
- Tous les 12 mois pour les systèmes non critiques
- Après toute mise à jour majeure ou migration d’environnement

Chaque test doit inclure :
- Vérification de l’intégrité des données restaurées
- Validation de la cohérence applicative
- Mesure du temps de restauration (RTO)
- Vérification du point de restauration (RPO)

## 11. Indicateurs de Performance (RTO/RPO)
Pour assurer la continuité d’activité, les objectifs suivants sont définis :

### 11.1 Recovery Time Objective (RTO)
- Critique : < 1 heure
- Important : < 4 heures
- Standard : < 24 heures

### 11.2 Recovery Point Objective (RPO)
- Critique : < 15 minutes
- Important : < 1 heure
- Standard : < 4 heures

Les équipes doivent constamment surveiller le respect de ces objectifs.

## 12. Journalisation et Traçabilité
Chaque opération de sauvegarde ou de restauration doit être enregistrée dans un registre :
- Heure de début
- Heure de fin
- Type de sauvegarde
- Statut (succès/échec)
- Opérateur responsable
- Logs d’erreur en cas d’échec

Les logs doivent être conservés 180 jours minimum.

## 13. Incidents de Sauvegarde
Les incidents suivants doivent déclencher une alerte automatique :
- Erreur de sauvegarde
- Fichier corrompu
- Incohérence de snapshot
- Perte de communication avec les serveurs de sauvegarde
- Espace disque insuffisant

Les incidents doivent être analysés dans un délai de 24h.

## 14. Révocation d’Accès aux Sauvegardes
Seules les personnes autorisées peuvent accéder aux sauvegardes :
- Administrateurs systèmes
- Équipe sécurité
- Responsables applicatifs (limité)

Toute tentative d’accès non autorisé doit déclencher une alerte.

## 15. Continuité d’Activité (BCP/PRA)
Les sauvegardes doivent intégrer le plan global de continuité :
- Documentation PRA à jour
- Serveur de secours opérationnel
- Tests réels de bascule (failover)
- Procédures de retour arrière (fallback)

## 16. Sanctions
Toute violation de la présente politique peut entraîner :
- Avertissement disciplinaire
- Suspension d’accès
- Sanction RH
- Licenciement en cas de négligence grave
- Poursuites légales en cas d’impact majeur

## 17. Révision de la Politique
Cette politique doit être révisée :
- Tous les 12 mois
- En cas d’incident majeur
- Après une mise à jour des systèmes
- Après une évolution légale ou réglementaire

