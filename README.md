
# 🤖 SVM Binary Classification & Hyperparameter Tuning

This project demonstrates how to use **Support Vector Machines (SVM)** for **binary classification** using a 2D synthetic dataset. It includes model training with **linear and RBF kernels**, **decision boundary visualization**, **hyperparameter tuning**, and **cross-validation**.

---

## 📁 Dataset

We use a **synthetic binary classification dataset** generated using `make_classification()` from `sklearn.datasets`. It contains **2 numerical features** and **binary class labels** (0 and 1).

---

## 🛠️ Tasks Performed

### 1. 📦 Import Libraries & Generate Dataset
```python
from sklearn import datasets
X, y = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
```

### 2. 🔍 Visualize Raw Data
```python
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
```

### 3. ✂️ Train-Test Split & Standardization
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 4. ⚙️ Train SVM with Linear Kernel
```python
from sklearn.svm import SVC
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train_scaled, y_train)
```

### 5. 📊 Visualize Linear Decision Boundary
```python
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train_scaled, y_train, clf=svm_linear)
```

### 6. 🌐 Train SVM with RBF Kernel
```python
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
plot_decision_regions(X_train_scaled, y_train, clf=svm_rbf)
```

### 7. 🔧 Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
```

### 8. 🧪 Cross-Validation & Evaluation
```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

best_model = grid.best_estimator_
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

y_pred = best_model.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## ✅ Final Output

```
Best Parameters: {'C': 10, 'gamma': 0.1}
Cross-validation scores: [0.92, 0.91, 0.93, 0.9, 0.94]
Mean CV Accuracy: 0.92
```

---

## 💡 Summary of SVM Workflow

| Step | Description |
|------|-------------|
| 1. Dataset | Create 2D binary dataset |
| 2. Preprocessing | Standardize features |
| 3. Linear SVM | Train & visualize |
| 4. RBF SVM | Train & visualize |
| 5. Hyperparameter Tuning | GridSearch on C & gamma |
| 6. Evaluation | Cross-validation + test |

---

## 👩‍💻 Tools Used

- Python
- Scikit-learn
- Matplotlib
- Mlxtend

---

## 📎 Notes

This project is ideal for:
- 🚀 Understanding kernel-based classification
- 📊 Visualizing decision boundaries
- 🎯 Tuning SVM parameters
- ✅ Practicing model evaluation with CV

