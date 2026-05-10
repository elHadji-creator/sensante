import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Encodeurs
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Features
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

# Sauvegarder les encodeurs et feature_cols
os.makedirs("models", exist_ok=True)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs sauvegardés !")
print(f"Sexe classes : {list(le_sexe.classes_)}")
print(f"Region classes : {list(le_region.classes_)}")
print(f"Features : {feature_cols}")