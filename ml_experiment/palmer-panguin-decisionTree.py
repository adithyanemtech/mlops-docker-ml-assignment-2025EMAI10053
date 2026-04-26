import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the Palmer Penguin dataset
# We use seaborn's built-in loader for convenience
penguins = sns.load_dataset('penguins')

# Drop rows with missing values for simplicity in this lab
penguins = penguins.dropna()

# 2. Preprocessing
# Encode categorical features and target
le = LabelEncoder()
penguins['species'] = le.fit_transform(penguins['species'])
penguins['island'] = le.fit_transform(penguins['island'])
penguins['sex'] = le.fit_transform(penguins['sex'])

X = penguins.drop('species', axis=1)
y = penguins['species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build a simple Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 5. Print Metrics
precision = precision_score(y_test, y_pred, average='weighted')
# AUC Score (multi-class requires 'ovr' or 'ovo' strategy)
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

print(f"Precision: {precision:.4f}")
print(f"AUC Score: {auc:.4f}")
