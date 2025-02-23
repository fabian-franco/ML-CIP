from transformation import Transformation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib

data = Transformation()

df = data.get_dataset()

X_balanced = df.drop('con_fin_', axis=1)
y_balanced = df['con_fin_']
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

model = LogisticRegression()

model.fit(X_train_balanced, y_train_balanced)

y_pred_balanced = model.predict(X_test_balanced)

accuracy_balanced = accuracy_score(y_test_balanced, y_pred_balanced)
report_balanced = classification_report(y_test_balanced, y_pred_balanced)

print(f"Accuracy con LogisticRegresion: {accuracy_balanced:.4f}")
print("Classification Report con LogisticRegresion:")
print(report_balanced)

cv_scores = cross_val_score(model, X_balanced, y_balanced, cv=5, scoring='accuracy')

print(f"Validación cruzada (5 folds) - Accuracy en cada fold: {cv_scores}")
print(f"Promedio de la validación cruzada: {np.mean(cv_scores):.4f}")

joblib.dump(model, 'models/lr_model.pkl')