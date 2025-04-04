
# Lab Report: Introduction to Deep Learning Using a Feed Forward Neural Network

**Course:** AI & Cybersecurity  
**Project:** Lab 1 – Feed Forward Neural Networks  
**Students:**  
- Alessandro Meneghini (s332228)  
- Pierfrancesco Elia (s331497)  
- Ankesh Porwal (s328746)  

---

## Task 1: Data Preprocessing

We began by importing the dataset from the provided CSV. To ensure data quality, we performed the following steps:

- Removed **duplicate entries**
- Removed **rows with NaN or Inf values**
- Dropped **irrelevant columns**, e.g., features with no variance across classes (e.g., `Fwd PSH Flags`)
- Explored the **distribution of each feature**, grouped by class label (Benign, PortScan, DoS Hulk, Brute Force)
- Identified and excluded features that were highly correlated with labels in a trivial way (e.g., `SYN Flag Count`)
- Split the data: **60% train**, **20% validation**, **20% test**
- Applied **standardization** using `StandardScaler` based on the training data

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

---

## Task 2: Shallow Neural Network

We trained a single-layer FFNN with various neuron configurations: **32**, **64**, and **128**, using the following setup:

- Activation: **Linear** (initial test), then switched to **ReLU**
- Optimizer: **AdamW**
- Loss Function: **CrossEntropyLoss**
- Batch Size: **64**
- Learning Rate: **0.0005**
- Epochs: **100** with **early stopping**

Performance was poor with **Linear activation**, significantly improved with **ReLU**.

```python
model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
)
```

We monitored the **loss curves** and selected the best model using **validation loss**. Test accuracy remained limited due to class imbalance and overfitting on dominant classes.

---

## Task 3: Feature Bias – Destination Port

We observed that **Brute Force attacks always targeted port 80**. To test this inductive bias:

- We **replaced port 80 with 8080** for Brute Force attacks in the test set
- Performance dropped, confirming the model was overfitting on the port feature

We then **removed the Destination Port** feature entirely and repeated preprocessing. Class balance changed notably:

- **PortScan** class instances reduced significantly due to duplicates with identical ports

To address class imbalance, we trained using **class weights** via `compute_class_weight` from `sklearn`.

```python
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights).float())
```

This improved the **F1 score** across minority classes and overall classification performance.

---

## Task 4: Deep Neural Network

We extended the architecture to **2–5 layers** using varying neuron configurations (e.g., 32-16, 64-32-16).

### Best Configuration:

- Layers: 3  
- Structure: `[64, 32, 16]`
- Activation: **ReLU**
- Batch Size: **64**
- Optimizer: **AdamW**

Validation and test accuracy improved significantly. The deeper network captured more complex patterns while maintaining generalization.

### Batch Size Impact

Tested batch sizes: 1, 32, 64, 128, 512. Results:

- **Small batches (1, 32)** led to noisy training but sometimes better generalization
- **Large batches (128, 512)** trained faster but were prone to overfitting

### Activation Function Impact

Compared **Linear**, **Sigmoid**, and **ReLU**:

- **ReLU** yielded the best results
- **Sigmoid** caused vanishing gradients, slower training
- **Linear** lacked non-linearity and resulted in underfitting

### Optimizer Impact

Compared:

- **SGD**
- **SGD with Momentum (0.1, 0.5, 0.9)**
- **AdamW**

**AdamW** consistently outperformed in convergence speed and final accuracy. Learning rate and epoch tuning confirmed the best configuration.

---

## Task 5: Overfitting and Regularization

We trained a deep FFNN with the following architecture:

- Layers: `[256, 128, 64, 32, 16]`
- Batch Size: **128**
- Optimizer: **AdamW**
- Epochs: **50**
- Regularization: **None initially**

### Observations:

- The model **overfitted** quickly (training loss ↓, val loss ↑)
- We introduced **Dropout** and **BatchNorm**, and used **weight decay** (AdamW)

```python
nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.5)
)
```

### Final Results:

- Dropout + BatchNorm significantly improved generalization
- Best performance achieved with **Dropout(0.5)** and **weight decay = 0.01**

---

## Conclusions

This lab introduced us to:

- Building FFNNs with PyTorch
- Handling data imbalance and bias
- Testing architectural variations (depth, batch size, activations, optimizers)
- Applying regularization to mitigate overfitting

The iterative approach helped refine our model and improve classification of minority classes, a key aspect in cybersecurity applications.
