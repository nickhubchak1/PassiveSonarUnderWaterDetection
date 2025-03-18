#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------
import subprocess
import sys

def install_requirements():
    # Check if requirements.txt exists
    try:
        with open('requirements.txt', 'r') as f:
            print("requirements.txt found, installing packages...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("All packages have been installed successfully!")
    except FileNotFoundError:
        print("requirements.txt not found. Please ensure the file is present.")

if __name__ == '__main__':
    install_requirements()