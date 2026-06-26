# =========================
# Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

sns.set(style='white')

# =========================
# Load Data
# =========================
dataset = pd.read_csv('iris.csv')

dataset.columns = [
    colname.strip(' (cm)').replace(" ", "_")
    for colname in dataset.columns
]

# =========================
# Feature Engineering
# =========================
dataset['sepal_length_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_length_width_ratio'] = dataset['petal_length'] / dataset['petal_width']

dataset = dataset[
    [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'sepal_length_width_ratio',
        'petal_length_width_ratio',
        'target'
    ]
]

# =========================
# Train / Test Split
# =========================
X = dataset.drop('target', axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=44,
    stratify=y
)

# =========================
# Logistic Regression (FIXED)
# =========================
logreg = LogisticRegression(
    C=0.0001,
    solver='lbfgs',
    max_iter=100
)

logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, pred_lr)

f1_lr = f1_score(y_test, pred_lr, average='micro')
prec_lr = precision_score(y_test, pred_lr, average='micro')
recall_lr = recall_score(y_test, pred_lr, average='micro')

train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100

# =========================
# Random Forest (CLASSIFIER – FIXED)
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=44
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

f1_rf = f1_score(y_test, pred_rf, average='micro')
prec_rf = precision_score(y_test, pred_rf, average='micro')
recall_rf = recall_score(y_test, pred_rf, average='micro')

train_acc_rf = rf.score(X_train, y_train) * 100
test_acc_rf = rf.score(X_test, y_test) * 100

# =========================
# Confusion Matrix Plot
# =========================
def plot_cm(cm, target_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('ConfusionMatrix.png', dpi=120)
    plt.show()

target_names = ['setosa', 'versicolor', 'virginica']
plot_cm(cm_lr, target_names, "Logistic Regression Confusion Matrix")

# =========================
# Feature Importance Plot
# =========================
importances = rf.feature_importances_
features = X.columns

fi_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=fi_df)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('FeatureImportance.png', dpi=120)
plt.show()

# =========================
# Save Scores
# =========================
with open('scores.txt', 'w') as f:
    f.write("Random Forest\n")
    f.write(f"Train Accuracy: {train_acc_rf:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc_rf:.2f}%\n")
    f.write(f"F1 Score: {f1_rf:.4f}\n")
    f.write(f"Recall: {recall_rf:.4f}\n")
    f.write(f"Precision: {prec_rf:.4f}\n\n")

    f.write("Logistic Regression\n")
    f.write(f"Train Accuracy: {train_acc_lr:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc_lr:.2f}%\n")
    f.write(f"F1 Score: {f1_lr:.4f}\n")
    f.write(f"Recall: {recall_lr:.4f}\n")
    f.write(f"Precision: {prec_lr:.4f}\n")


