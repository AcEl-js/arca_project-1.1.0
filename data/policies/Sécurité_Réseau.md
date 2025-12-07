# Politique de Sécurité Réseau 

## 1. Objet
La présente politique vise à définir les exigences, bonnes pratiques et obligations nécessaires pour garantir la sécurité, la résilience et la disponibilité du réseau interne de l’entreprise. Elle établit les règles relatives au contrôle des connexions, au filtrage, à l’accès distant, à la segmentation, à la surveillance et à la prévention des intrusions.

## 2. Portée
Cette politique couvre :
- Les réseaux internes (LAN, VLAN)
- Les réseaux distants (WAN, VPN)
- Les réseaux cloud
- Les pare-feux, routeurs, switches, proxies
- Les serveurs de communication
- Les devices connectés (PC, mobile, IoT interne)
- Les applications web internes et externes

Elle s’applique à tout employé, prestataire ou tiers disposant d’un accès au réseau.

## 3. Principes Directeurs
Les principes suivants doivent être respectés :
- **Sécurité par défaut** : tout service non utilisé doit être désactivé.
- **Principe du moindre privilège** : accès strictement limité.
- **Segmentation** : isoler les zones critiques.
- **Surveillance continue** : monitorer en temps réel.
- **Tolérance zéro** pour les accès non autorisés.

## 4. Architecture Réseau
L’architecture doit inclure :
- Un réseau segmenté par rôles (RH, Finance, Dev, Prod)
- Une DMZ dédiée aux services publics
- Un firewall principal filtrant tout le trafic
- Un IDS/IPS analysant les paquets en temps réel
- Des VLAN séparés pour les devices IoT ou invités

Chaque composant doit être documenté dans le schéma réseau officiel.

## 5. Gestion des Pare-feux
Les pare-feux doivent respecter :
- Une configuration basée sur des règles “deny-all” par défaut
- Une ouverture des ports strictement nécessaire
- Une révision mensuelle des règles
- Une documentation de chaque modification
- Une alerte automatique en cas de tentative suspecte

Toute règle non utilisée doit être supprimée.

## 6. Contrôle des Ports et Protocoles
Le réseau doit respecter :
- Fermeture de tous les ports non utilisés
- Interdiction de protocoles obsolètes (FTP, Telnet)
- Obligation d’utiliser SSH, SFTP, HTTPS
- Vérification continue des ports ouverts
- Blocage des scans réseau suspects

Des audits réguliers doivent être effectués via des outils automatisés.

## 7. Sécurité WiFi
Les règles suivantes s'appliquent :
- WPA3 obligatoire
- Interdiction d’utiliser des WiFi publics pour le travail
- Réseaux invités isolés du réseau principal
- Mot de passe WiFi changé tous les 3 mois
- Filtrage MAC pour les appareils sensibles

Tout appareil non approuvé doit être bloqué.

## 8. Accès Distant et VPN
L’accès distant doit respecter :
- VPN obligatoire avec chiffrement AES-256
- MFA obligatoire pour toute connexion externe
- Interdiction d’utiliser un réseau non sécurisé
- Déconnexion automatique après 15 minutes d’inactivité
- Journalisation de toutes les connexions distantes

Toute tentative de connexion échouée doit déclencher une alerte.

## 9. Sécurité des Serveurs et Services Exposés
Les serveurs accessibles depuis l’extérieur doivent :
- Être isolés dans une DMZ
- Utiliser HTTPS obligatoirement
- Avoir des certificats TLS valides
- Être patchés régulièrement
- Être surveillés 24/7 par les outils de monitoring réseau

Les services non utilisés doivent être désactivés immédiatement.

## 10. Contrôle des Adresses IP
Les règles suivantes s’appliquent :
- Attribution d’adresses IP statiques pour les systèmes critiques
- DHCP seulement pour les postes standards
- Interdiction d’IP non documentée
- Surveillance des IP anormales (scans, flood, ARP spoofing)

## 11. Surveillance Réseau
Le réseau doit être surveillé à l’aide de :
- SIEM
- IDS/IPS
- Analyse comportementale (UEBA)
- Systèmes d’alerte automatique
- Dashboards temps réel

Tout comportement suspect doit être investigué immédiatement.

## 12. Gestion des Logs Réseau
Les logs doivent inclure :
- Tentatives d’accès
- Connexions entrantes/sortantes
- Modifications sur les pare-feux
- Alertes IDS/IPS
- Détection d’anomalies

#### Durée de conservation
- Logs standards : 90 jours
- Logs critiques : 180 jours

Les logs doivent être stockés dans un système centralisé et chiffré.

## 13. Protection Contre les Attaques
Le réseau doit être protégé contre :
- DDoS
- MITM (Man-In-The-Middle)
- Spoofing
- Sniffing
- SQL injection (via WAF)
- Malware propagation

Des solutions doivent être mises en place :
- WAF pour applications web
- Antivirus/EDR pour endpoints
- Anti-DDoS layer 3/4

## 14. Patch Management Réseau
Les équipements réseau doivent être :
- Patchés mensuellement
- Testés avant mise en production
- Audités après chaque mise à jour majeure

Aucun équipement obsolète ne doit rester connecté.

## 15. Accès Physiques au Matériel Réseau
L’accès physique doit être :
- Strictement limité aux administrateurs réseau
- Contrôlé par badges ou accès biométriques
- Journalisé automatiquement
- Sous vidéosurveillance continue

Aucun câble ou switch ne doit être accessible par le personnel non autorisé.

## 16. Gestion des Incidents Réseau
En cas d'incident réseau :
- Le service IT doit intervenir immédiatement
- Le système affecté doit être isolé
- Un diagnostic doit être réalisé dans les 24h
- Les actions correctives doivent être documentées
- Un rapport final doit être produit

Toute tentative d’intrusion doit être reportée au RSSI.

## 17. Sanctions
Toute violation de cette politique peut entraîner :
- Suspension d’accès
- Avertissement officiel
- Sanction disciplinaire
- Licenciement en cas de compromission volontaire
- Procédure judiciaire en cas d’attaque interne

## 18. Révision
Cette politique doit être révisée tous les 12 mois ou après un incident majeur.


