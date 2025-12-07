# Politique de Sécurité Informatique

## 1. Objet
La présente politique a pour objectif d’établir un cadre complet, structuré et uniforme pour la protection des systèmes d’information, incluant la confidentialité, l’intégrité, la disponibilité, la résilience, la traçabilité et la gouvernance des données numériques. Cette politique s’applique à l’ensemble du personnel, y compris les salariés, prestataires, consultants, partenaires et toute personne ayant accès aux systèmes de l'entreprise.

## 2. Authentification et Accès
- L’authentification multifacteur (MFA) est obligatoire pour tous les utilisateurs internes et externes.
- Tous les mots de passe doivent comporter au moins 12 caractères, incluant majuscules, minuscules, chiffres et caractères spéciaux.
- Le renouvellement des mots de passe est obligatoire tous les 90 jours.
- Les comptes inactifs pendant 30 jours doivent être automatiquement désactivés.
- L'accès aux environnements sensibles nécessite une validation hiérarchique.

## 3. Sécurité des Postes de Travail
- Tous les postes (PC, laptops, tablettes) doivent être chiffrés avec un chiffrement AES-256.
- Installation de logiciels non validés interdite.
- Les mises à jour critiques doivent être appliquées dans les 72 heures suivant leur publication.
- Le verrouillage automatique doit être activé après 5 minutes d’inactivité.

## 4. Réseaux et Accès
- L’accès distant aux systèmes internes est strictement limité au VPN approuvé.
- Le réseau interne doit être segmenté en zones : DMZ, Production, Administration, Développement.
- Les pare-feux doivent appliquer des règles “deny-all except allowed”.
- Le trafic réseau doit être inspecté en continu par des outils IDS/IPS (Intrusion Detection/Prevention Systems).

## 5. Journalisation
- Tous les événements critiques (connexion, erreur, modification système, accès admin) doivent être journalisés.
- La durée minimale de conservation des logs est fixée à 180 jours.
- Les journaux administratifs doivent être stockés dans un coffre-fort numérique.

## 6. Sécurité Applicative
- Toutes les applications doivent être développées conformément aux standards OWASP.
- Les scans SAST et DAST doivent être réalisés avant chaque mise en production.
- Les clés API doivent être régénérées au minimum tous les 12 mois.
- Les composants open-source doivent être vérifiés via un scanner SCA.

## 7. Protection Anti-Malware
- Tous les postes doivent être équipés d’un antivirus à jour.
- Analyse complète obligatoire chaque semaine.
- Les fichiers téléchargés doivent obligatoirement passer par un sandboxing automatique.

## 8. Sécurité Physique
- Les salles serveurs doivent être protégées par badge, biométrie ou double contrôle.
- Les caméras doivent enregistrer en continu, avec une rétention minimale de 30 jours.
- Les visiteurs doivent être escortés et inscrits dans le registre de sécurité.

## 9. Sensibilisation
- Chaque employé doit suivre une formation cybersécurité annuelle.
- Les tests de phishing internes doivent être réalisés chaque trimestre.
- Les employés échouant deux tests consécutifs doivent suivre une formation renforcée.

## 10. Gestion des Vulnérabilités
- Les scans d'infrastructure doivent être réalisés au moins une fois par mois.
- Les vulnérabilités critiques doivent être corrigées en moins de 7 jours.
- Les vulnérabilités élevées doivent être corrigées en moins de 30 jours.

## 11. Gestion des Incidents
- Tout incident doit être déclaré dans l'heure.
- L'équipe de réponse doit analyser l’incident dans les 24 heures.
- Un rapport post-incident doit être rédigé dans les 48 heures.

## 12. Gouvernance des Données
- Les données doivent être classifiées selon une grille interne : Publique, Interne, Confidentielle, Très Confidentielle.
- Le stockage des données sensibles doit être chiffré au repos et en transit.
- Les sauvegardes doivent être testées au moins une fois tous les six mois.

## 13. Contrôles d’Accès
- Principe du moindre privilège obligatoire.
- Les accès admin doivent être revus tous les 15 jours.
- Les comptes partagés sont strictement interdits.

## 14. Continuité d’Activité
- Un plan de crise doit être maintenu.
- Un PRA (Plan de Reprise d’Activité) doit être testé chaque année.
- Un PCA (Plan de Continuité d’Activité) doit garantir la reprise en moins de 4 heures.

## 15. Conformité
- Cette politique doit être conforme aux normes ISO 27001, ISO 22301 et NIST SP 800-53.
- Tout employé doit signer une charte d’utilisation du SI.
- Toute violation de cette politique peut donner lieu à des sanctions disciplinaires.

## 16. Révision
Cette politique doit être réévaluée chaque 12 mois par le comité de sécurité.

