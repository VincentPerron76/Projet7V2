#!/bin/bash

# Mise à jour des paquets
sudo apt-get update

# Installation de python3-distutils
sudo apt-get install -y python3-venv
pip install setuptools

# Autres dépendances système que vous pourriez avoir besoin d'ajouter
# sudo apt-get install -y <autre_paquet>
