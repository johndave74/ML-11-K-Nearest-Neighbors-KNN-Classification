# ML-11-K-Nearest-Neighbors-KNN-Classification
This project explores the use of the K-Nearest Neighbors (KNN) algorithm on a classified dataset. The objective is to build a predictive model that can classify new observations based on a set of hidden features.

# 📘 K-Nearest Neighbors (KNN) Classification

This project explores the use of the **K-Nearest Neighbors (KNN)** algorithm on a classified dataset. The objective is to build a predictive model that can classify new observations based on a set of hidden features.

## 📂 Dataset Description

The dataset is provided in `Classified Data.txt`, with feature names anonymized (e.g., `WTT`, `PTI`, `EQW`, etc.) and includes a `TARGET CLASS` column representing the binary classification output.

### Sample Data

| WTT    | PTI    | EQW    | SBI    | ... | NXJ    | TARGET CLASS |
|--------|--------|--------|--------|-----|--------|---------------|
| 0.914  | 1.162  | 0.568  | 0.755  | ... | 1.231  | 1             |
| 0.636  | 1.004  | 0.535  | 0.826  | ... | 1.493  | 0             |
| ...    | ...    | ...    | ...    | ... | ...    | ...           |

---

## 📊 Data Preprocessing

- **Standardization**: Used `StandardScaler` to normalize feature values.
- **Train-Test Split**: Dataset split into 70% training and 30% testing.
- **Feature Engineering**: No additional features were engineered since the focus is on modeling with given features.

---

## 🔍 Exploratory Data Analysis

The dataset was explored using:
- `info()`, `describe()` for structure and summary.
- `sns.pairplot()` and `sns.heatmap()` for pattern detection and feature correlation.

---

## 🧠 Model: K-Nearest Neighbors (KNN)

KNN is a **non-parametric, instance-based learning** algorithm that classifies data points based on the class of their nearest neighbors in the feature space.

### Model Steps:

1. **Model Instantiation** with default `k=1`
2. **Training** the model with the training data
3. **Prediction** on the test set
4. **Evaluation** using classification report and confusion matrix
5. **Hyperparameter Tuning**: Tested multiple values of `k` to determine the best-performing one.

### Performance Evaluation

| k (Neighbors) | Accuracy Score |
|---------------|----------------|
| 1             | Lower accuracy (high variance) |
| 5             | Improved generalization |
| 17 (Optimal)  | Best test accuracy observed |

Used the Elbow Method to visualize error rate vs. `k`:

```python
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```

---

## 📈 Final Model

- **Algorithm**: KNN
- **Best Parameter**: `n_neighbors = 23`
- **Accuracy**: Improved after tuning
- **Insights**:
  - Initial model with `k=1` led to overfitting.
  - Increasing `k` reduced error up to a point.
  - Performance stabilizes after `k=23`.

---

## 🧾 Conclusion

The KNN model, while simple and intuitive, demonstrated effective classification capabilities when tuned correctly. Feature scaling and careful selection of `k` significantly impacted performance.

---

## 📁 Files in Repository

| File                         | Description                         |
|------------------------------|-------------------------------------|
| `01-K Nearest Neighbors.ipynb` | Jupyter notebook with code and visuals |
| `Classified Data.txt`         | Dataset used for training/testing   |
| `README.md`                   | Project overview and insights       |

---

## 🚀 Future Improvements

- Apply cross-validation for better performance estimation.
- Try other distance metrics (Manhattan, Cosine).
- Compare with other classifiers (SVM, Logistic Regression, Decision Tree).

---
