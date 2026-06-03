import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('ml/banana_leaf_dataset.csv')  # update path

# Encode categorical columns
le_texture = LabelEncoder()
le_color = LabelEncoder()
le_soil = LabelEncoder()

df['Texture']          = le_texture.fit_transform(df['Texture'])
df['ColorIntensity']   = le_color.fit_transform(df['ColorIntensity'])
df['SoilType']         = le_soil.fit_transform(df['SoilType'])

X = df[['LeafLength(cm)', 'LeafWidth(cm)', 'ColorIntensity',
        'SpotsPresent', 'MoistureLevel(%)', 'Texture',
        'Humidity(%)', 'Temperature(°C)', 'SoilType']]
y = df['DiseaseLabel']  # update to your target column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'decision_tree.pkl':        DecisionTreeClassifier(),
    'random_forest.pkl':        RandomForestClassifier(),
    'gradient_boosting.pkl':    GradientBoostingClassifier(),
    'svm.pkl':                  SVC(),
    'knn.pkl':                  KNeighborsClassifier(),
    'logistic_regression.pkl':  LogisticRegression(max_iter=1000),
}

for filename, model in models.items():
    model.fit(X_train, y_train)
    with open(f'ml_models/{filename}', 'wb') as f:  # update path
        pickle.dump(model, f)
    print(f"✅ Saved {filename}")

print("All models retrained and saved!")