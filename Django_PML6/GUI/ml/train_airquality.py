import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models_aq')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load your real dataset ─────────────────────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, 'airquality_dataset.csv'))

print("✅ Dataset loaded! Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Unique Types:", df['Type'].unique())

# ── Drop unused columns ────────────────────────────────────────
df = df.drop(columns=['Date', 'City'], errors='ignore')

# ── Encode target label ────────────────────────────────────────
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])  # Industrial=0, Residential=1
joblib.dump(le, os.path.join(SAVE_DIR, 'label_encoder_aq.pkl'))

print("Encoded classes:", le.classes_)

# ── Features & Target ──────────────────────────────────────────
X = df[['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']]
y = df['Type']

print(f"Samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Classifiers ────────────────────────────────────────────────
classifiers = {
    'Logistic Regression':    LogisticRegression(max_iter=1000),
    'Decision Tree':          DecisionTreeClassifier(),
    'Random Forest':          RandomForestClassifier(n_estimators=100),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors':    KNeighborsClassifier(),
    'Naive Bayes':            GaussianNB(),
}

filenames = {
    'Logistic Regression':    'lr_aq.pkl',
    'Decision Tree':          'dt_aq.pkl',
    'Random Forest':          'rf_aq.pkl',
    'Support Vector Machine': 'svm_aq.pkl',
    'K-Nearest Neighbors':    'knn_aq.pkl',
    'Naive Bayes':            'nb_aq.pkl',
}

accuracies = {}
print("\n🌬️ Training Air Quality Models...\n")

for name, clf in classifiers.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', clf),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    joblib.dump(pipeline, os.path.join(SAVE_DIR, filenames[name]))
    print(f"  ✅ {name:<30} Accuracy: {acc*100:.2f}%")

joblib.dump(accuracies, os.path.join(SAVE_DIR, 'accuracies_aq.pkl'))
print("\n✅ All models saved to ml/saved_models_aq/\n")