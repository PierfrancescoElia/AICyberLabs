
# Lab Report: Introduction to Deep Learning Using a Feed Forward Neural Network

**Course:** AI & Cybersecurity  
**Project:** Lab 1 – Feed Forward Neural Networks  
**Students:**  
- Alessandro Meneghini (s332228)  
- Pierfrancesco Elia (s331497)  
- Ankesh Porwal (s328746)  

---

## Task 1: Data Preprocessing

The dataset from CICIDS2017 was preprocessed by removing noisy and irrelevant entries:

- Dropped duplicates and rows with NaN or infinite values.
- Visualized feature distributions across classes to identify features with trivial correlation (e.g., flags).
- Removed irrelevant features (e.g., `Fwd PSH Flags`) when they provided trivial separation.
- Normalized using `StandardScaler` and split into training (60%), validation (20%), and test (20%).

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

---

## Task 2: Shallow Neural Network

Implemented a single-layer FFNN with varying neuron sizes:

- Neurons per layer: **32, 64, 128**
- Batch size: **64**, Epochs: **100**
- Activation: **Linear → ReLU**
- Optimizer: **AdamW**
- Loss: **CrossEntropy**

We observed overfitting with Linear, and better generalization with ReLU.

```python
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, num_classes)
)
```

Model selection was based on validation accuracy. Test F1-scores were poor due to class imbalance.

---

## Task 3: Impact of Specific Features

### 🔹 Port as Bias Feature
All Brute Force attacks occurred on **port 80**. We tested model generalization by changing ports to 8080 in the test set. The accuracy dropped, showing overfitting to this port.

### 🔹 Removing Port Feature
After dropping the port feature:
- Significant drop in **PortScan** instances due to duplicates
- Dataset became more balanced
- Training repeated with best model from previous task

### 🔹 Weighted Loss
To handle imbalance:
```python
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).float())
```

Improved F1-scores, especially for rare classes like **Brute Force**.

---

## Task 4: Deep Neural Network

Tested deeper FFNNs with 2–5 layers and hidden units: `[16, 8, 4, 2]`, etc.

- Activation: **ReLU**
- Optimizer: **AdamW**
- Batch size: **64**
- Best model had **3 layers**: `[32, 16, 8]`

### 🔸 Impact of Batch Size
- Small batches led to more noise but better generalization
- Larger batches trained faster but slightly overfit

### 🔸 Activation Functions
Compared **ReLU**, **Sigmoid**, **Linear**:

- **ReLU** performed best due to non-linearity and gradient stability.
- **Sigmoid** slowed down training.
- **Linear** performed poorly.

### 🔸 Optimizer Impact
Compared:
- **SGD**
- **SGD + Momentum (0.1, 0.5, 0.9)**
- **AdamW**

AdamW was most efficient and achieved best convergence. SGD required more tuning.

---

## Task 5: Overfitting and Regularization

A deeper model `[256, 128, 64, 32, 16]` was trained:

- Overfit without regularization (val loss ↑, train loss ↓)
- Added **Dropout(0.5)** and **BatchNorm**
- Used **weight_decay=0.01** in AdamW

```python
nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.5)
)
```

Regularization significantly improved validation accuracy and prevented overfitting.

---

## Conclusion

This lab demonstrated:

- Preprocessing steps to ensure clean training data
- Limitations of shallow networks and advantages of deeper FFNNs
- How dataset features and imbalance affect generalization
- The role of batch size, activation function, optimizer, and regularization in improving performance

The final model was robust and achieved good classification scores across all classes, including rare ones.

