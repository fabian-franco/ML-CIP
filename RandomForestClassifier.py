import numpy as np
from transformation import Transformation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib

data = Transformation()

df = data.get_dataset()

X_balanced = df.drop('con_fin_', axis=1)
y_balanced = df['con_fin_']
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train_balanced, y_train_balanced)

y_pred_rf = rf_model.predict(X_test_balanced)

accuracy_rf = accuracy_score(y_test_balanced, y_pred_rf)
report_rf = classification_report(y_test_balanced, y_pred_rf)

print(f"Accuracy con Random Forest: {accuracy_rf:.4f}")
print("Classification Report con Random Forest:")
print(report_rf)

cv_scores = cross_val_score(rf_model, X_balanced, y_balanced, cv=5, scoring='accuracy')

print(f"Validación cruzada (5 folds) - Accuracy en cada fold: {cv_scores}")
print(f"Promedio de la validación cruzada: {np.mean(cv_scores):.4f}")
print(f"Desviación estándar de la validación cruzada: {np.std(cv_scores):.4f}")

joblib.dump(rf_model, 'models/rf_model.pkl')