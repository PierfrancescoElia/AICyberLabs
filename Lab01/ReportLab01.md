
# Lab Report: Introduction to Deep Learning Using a Feed Forward Neural Network

**Course:** AI & Cybersecurity  
**Project:** Lab 1 – Feed Forward Neural Networks  
**Students:**  
- Alessandro Meneghini (s332228)  
- Pierfrancesco Elia (s331497)  
- Ankesh Porwal (s328746)

## 1. Overview

The main goal of this lab was to explore how Feed Forward Neural Networks (FFNNs) can be applied to cybersecurity problems, specifically for intrusion detection. We worked with the CICIDS2017 dataset, a well-known benchmark in network security that includes a mix of normal and malicious traffic. Our task was to build and evaluate models capable of classifying different types of network behavior using PyTorch.

We began with preprocessing the dataset by cleaning missing values and duplicates, selecting key features, applying normalization, and encoding labels. The data was then split into training, validation, and test sets. Special care was taken to ensure consistent preprocessing across all splits to avoid data leakage.

Our first experiments involved building a shallow FFNN with one hidden layer. Initially, the model performed poorly with a Linear activation function. When we switched to ReLU, performance improved significantly. This highlighted the importance of non-linearity in learning complex patterns. Early stopping was used to select the best-performing model based on validation loss.

We then explored how the model was influenced by specific features, especially `Destination Port`. We found that the model heavily relied on this field to detect Brute Force attacks. When we changed the port in the test set, performance dropped dramatically. This indicated overfitting to that feature. To resolve this, we removed the port field and retrained the model, this time also introducing class-weighted loss functions to address class imbalance.

Next, we expanded the architecture to deeper FFNNs with multiple hidden layers. These models achieved better generalization, but also showed signs of overfitting as complexity increased. We applied regularization techniques such as dropout, batch normalization, and weight decay to address this issue. These adjustments improved validation and test performance across the board.

Finally, we tested different batch sizes, activation functions, and optimizers. ReLU consistently provided the best results, and AdamW was the most effective optimizer in terms of both accuracy and training speed.

This lab helped us understand not only how to build neural networks, but also how to debug and improve them through careful analysis, tuning, and regularization — all within a realistic cybersecurity scenario.


---

## Task 1: Data Preprocessing

The first step in building a reliable machine learning pipeline was the preprocessing of the CICIDS2017 dataset. This dataset includes labeled network traffic records representing both benign activity and different types of attacks. Proper preprocessing was essential to ensure model stability, accuracy, and generalizability.

### 1.1 Data Cleaning

We began by removing all rows containing missing (`NaN`) values and duplicate entries to eliminate inconsistencies and redundancy. This helped improve data quality and prevented misleading learning signals.

### 1.2 Feature Selection

A subset of relevant numerical features was selected based on domain knowledge and the instructions provided. These features include metrics such as `Flow Duration`, `Flow IAT Mean`, `Bwd Packet Length Mean`, `Flow Bytes/s`, `SYN Flag Count`, among others. Categorical and highly sparse features were excluded to focus on numerical input suited for neural networks.

### 1.3 Normalization

Since the selected features had different ranges and distributions, we applied z-score normalization using `StandardScaler` from scikit-learn. This step ensured that all input features contributed equally to the learning process and avoided domination by high-magnitude variables.

Normalization was fit only on the training set and then applied to both the validation and test sets to avoid information leakage and ensure proper generalization.

### 1.4 Label Encoding

The dataset's target labels (`Benign`, `DoS Hulk`, `PortScan`, `Brute Force`) were encoded as integer classes using label encoding. This transformation is necessary for the model to compute the cross-entropy loss correctly.

### 1.5 Train/Validation/Test Split

The dataset was split into three parts:
- **60% for training**
- **20% for validation**
- **20% for testing**

The split was stratified to maintain the original class distribution across all partitions. This structure allowed us to monitor model performance on unseen data and apply early stopping techniques during training.

### 1.6 Outlier Handling

Visual inspection of the feature distributions revealed the presence of significant outliers, particularly in flow-based features. These were not removed directly, but normalization helped reduce their impact. Additional techniques, such as robust scaling or clipping, were considered but not adopted in the final implementation due to satisfactory model performance.

---

## Task 2: Shallow Neural Network

The second task focused on developing a baseline classification model using a shallow Feed Forward Neural Network (FFNN). This step helped us understand how a minimal neural architecture performs on the intrusion detection task before experimenting with deeper networks.

### 2.1 Model Architecture

We implemented a shallow FFNN with a single hidden layer. The number of neurons in this layer was varied across three configurations: 32, 64, and 128. The output layer used a softmax activation (through `CrossEntropyLoss`) to classify samples into four categories: Benign, PortScan, DoS Hulk, and Brute Force.

The initial activation function used for the hidden layer was **Linear**, as instructed. However, due to its inability to introduce non-linearity, the model showed poor performance, particularly in classifying attack types.

### 2.2 Training Setup

- **Loss function**: CrossEntropyLoss  
- **Optimizer**: AdamW  
- **Learning rate**: 0.0005  
- **Batch size**: 64  
- **Epochs**: Up to 100 with early stopping  
- **Weight initialization**: Default (PyTorch)

Early stopping was applied based on the validation loss, with a patience threshold to prevent overfitting.

### 2.3 Evaluation Metrics

We monitored both training and validation loss during training and evaluated performance using:
- Overall accuracy
- Per-class F1-score
- Confusion matrices (in the notebook)

These metrics helped assess how well the model generalized to unseen data and handled class imbalance.

### 2.4 Results and Observations

The model trained with a **Linear activation** struggled to learn non-linear decision boundaries and underfit the data, regardless of the number of neurons. When we switched the activation function to **ReLU**, performance improved significantly across all configurations. The ReLU-based models were better at separating the classes and captured more complex patterns in the feature space.

Among the three tested configurations, the network with **128 neurons** in the hidden layer and ReLU activation achieved the best results in terms of validation and test accuracy. This configuration was selected as the baseline for further experiments.

### 2.5 Limitations

- The shallow architecture lacks the capacity to model highly non-linear patterns, limiting its accuracy.
- Without regularization or feature filtering, some noise in the dataset impacted the model's stability during training.
- The model performed poorly on minority classes (especially Brute Force), which highlighted the need for better class balancing techniques in later stages.

---

## Task 3: The Impact of Specific Features

In this task, we investigated how the presence and distribution of specific features—particularly `Destination Port`—influenced the model’s learning behavior and generalization capabilities. We also explored how the dataset's inherent biases affected classification performance and how these biases could be mitigated.

### 3.1 Feature Bias: The Case of Destination Port

During initial analysis, we noticed that all Brute Force attacks in the dataset occurred on port 80. As a result, the model learned to associate the Brute Force class directly with this port, instead of identifying more general traffic behavior.

To test this hypothesis, we altered the test set by changing the destination port for Brute Force attacks from 80 to 8080. When we ran inference using the previously trained model, the ability to correctly classify Brute Force samples dropped significantly. This confirmed that the model had overfit to a specific feature value rather than learning meaningful patterns from the rest of the data.

### 3.2 Feature Removal and Reprocessing

To eliminate this bias, we removed the `Destination Port` feature entirely and repeated the full preprocessing pipeline:
- Dropped NaNs and duplicates again.
- Re-applied normalization.
- Performed label encoding and data splitting.

Interestingly, the number of **PortScan** samples decreased significantly after this step. This happened because many PortScan entries had been differentiated only by their destination port, and removing that feature caused many to be identified as duplicates and removed. As a result, class imbalance became more severe, with minority classes becoming underrepresented.

### 3.3 Weighted Loss for Class Imbalance

To address the new imbalance, we implemented a **weighted loss function**. We used `sklearn.utils.class_weight.compute_class_weight` with the `balanced` option to compute class weights automatically. These weights were then passed to the `CrossEntropyLoss` function in PyTorch.

This change led to a clear improvement in classification performance, particularly for the minority classes like **Brute Force** and **PortScan**, which had previously been underrepresented. F1-scores for these classes improved, although some drop in overall accuracy was observed—an expected trade-off when optimizing for balanced per-class performance.

### 3.4 Model Retraining and Evaluation

We retrained the best-performing shallow model (128 neurons, ReLU) using the new dataset (without the port feature) and the class-weighted loss. While the model's performance on the majority class (Benign) slightly declined, the ability to detect rare attack types improved meaningfully. This balanced behavior is more appropriate for real-world intrusion detection systems, where rare events must not be ignored.

---

## Task 4: Deep Neural Network

In this task, we extended our model architecture from a shallow neural network to a deeper Feed Forward Neural Network (FFNN), in order to capture more complex patterns and improve classification performance. We also systematically tested the effects of different hyperparameters and architectural choices, including batch size, activation function, and optimizer.

### 4.1 Deep Network Design

We experimented with networks composed of 2 to 5 hidden layers. The number of neurons per layer varied across configurations, using values such as 2, 4, 8, 16, and 32. All hidden layers used the ReLU activation function unless otherwise stated. The output layer remained unchanged, using softmax via `CrossEntropyLoss` for multiclass classification.

Early stopping was applied in all cases to prevent overfitting, and model selection was based on validation loss.

The best-performing deep network had three hidden layers with 32, 16, and 8 neurons respectively. This configuration provided a good balance between model complexity and generalization.

### 4.2 Effect of Batch Size

We evaluated the model using different batch sizes: 1, 32, 64, 128, and 512.

- **Small batch sizes** (e.g., 1 or 32) produced noisy loss curves but often generalized better.
- **Larger batch sizes** (e.g., 128 or 512) trained faster and more smoothly but were prone to overfitting and showed lower validation performance.

A batch size of **64** was identified as the best compromise between stability, speed, and generalization.

### 4.3 Effect of Activation Function

We compared three activation functions in the hidden layers: Linear, Sigmoid, and ReLU.

- **Linear** activations led to underfitting and poor class separation.
- **Sigmoid** activations caused vanishing gradients and slower training.
- **ReLU** offered the best training speed and classification accuracy.

ReLU was therefore retained as the default choice for deeper networks.

### 4.4 Effect of Optimizer

We tested several optimizers:
- Stochastic Gradient Descent (SGD)
- SGD with Momentum (0.1, 0.5, 0.9)
- AdamW

AdamW consistently delivered better results in terms of convergence speed and final accuracy. Although SGD with momentum improved performance compared to vanilla SGD, it was still less effective than AdamW. This confirmed AdamW as the most suitable optimizer for our task.

### 4.5 Final Observations

Deeper architectures significantly improved the model’s ability to capture non-linear and subtle patterns in the data. However, as complexity increased, the risk of overfitting became more evident. These findings motivated the use of regularization techniques in the next task.


---

## Task 5: Overfitting and Regularization

The final task focused on identifying overfitting in deep neural networks and applying regularization strategies to improve generalization. We specifically worked with an overparameterized Feed Forward Neural Network (FFNN) and assessed how dropout, batch normalization, and weight decay could mitigate overfitting effects.

### 5.1 Overparameterized Network

We designed a deep FFNN with the following architecture:

- **Hidden layers**: 6
- **Neurons per layer**: [256, 128, 64, 32, 16]
- **Activation**: ReLU
- **Batch size**: 128
- **Epochs**: 50 (with early stopping)
- **Loss function**: CrossEntropy
- **Optimizer**: AdamW (lr = 0.0005)

This model had sufficient capacity to fit the training data very well. However, validation loss plateaued early while training loss continued to decrease — a clear sign of overfitting. The test accuracy was also lower than expected, confirming poor generalization.

### 5.2 Applying Regularization

To reduce overfitting, we applied three key regularization techniques:

#### a) Dropout

We added **dropout layers** after each hidden layer, initially testing with a dropout rate of 0.3. This forced the model to learn redundant and more generalizable features by randomly deactivating neurons during training. The addition of dropout helped improve validation performance and reduced overfitting.

#### b) Batch Normalization

Batch normalization layers were inserted before the activation functions. This stabilized the learning process by reducing internal covariate shift and helped the network converge faster. Models with batch normalization exhibited smoother loss curves and more stable accuracy across epochs.

#### c) Weight Decay

We introduced **weight decay** (L2 regularization) by configuring the `weight_decay` parameter in the AdamW optimizer. This penalized large weights and discouraged the network from relying too heavily on any single neuron. Weight decay further improved the validation loss and test F1-scores.

### 5.3 Combined Impact

The combination of all three techniques—dropout, batch normalization, and weight decay—provided the best results. The network achieved improved generalization, better F1-scores on minority classes, and more consistent performance on the test set.

Validation and test losses were notably more aligned, and the training process became more stable. While the overall accuracy did not increase dramatically, the model’s robustness to overfitting and performance on rare classes improved meaningfully.
