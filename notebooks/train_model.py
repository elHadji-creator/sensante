import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")

print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")
from sklearn.preprocessing import LabelEncoder

# Encoder
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Features et cible
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee', 'region_encoded']
X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

print("Modele sauvegarde dans models/model.pkl")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print("Matrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.savefig("figures/confusion_matrix.png")
plt.show()

# === TEST DU MODELE SERIALISE ===
model_loaded = joblib.load("models/model.pkl")

# Nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'frissons': True,
    'nausee': True,
    'region': 'Dakar'
}

# Encoder
sexe_enc = le_sexe.transform([nouveau_patient['sexe']])[0]
region_enc = le_region.transform([nouveau_patient['region']])[0]

# Features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    int(nouveau_patient['frissons']),
    int(nouveau_patient['nausee']),
    region_enc
]

# Prédiction
diagnostic = model_loaded.predict([features])[0]

print("\n--- RESULTAT DU TEST ---")
print(f"Diagnostic prédit : {diagnostic}")

print("\n--- IMPORTANCE DES FEATURES ---")
importances = model.feature_importances_

for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1],
                        reverse=True):
    print(f"{name:20s} : {imp:.3f}")