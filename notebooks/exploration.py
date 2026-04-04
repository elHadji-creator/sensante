import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("data/patients_dakar.csv")

# Afficher les premières lignes
print(df.head())

# Nombre total de patients
print("Nombre de patients :", len(df))