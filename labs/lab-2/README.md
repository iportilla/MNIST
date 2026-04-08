# Supervised Learning: Image Classification

> **Audience:** CS students  
> **Example domain:** Classifying images as `Dog` or `Cat`

---

## What is Supervised Learning?

Supervised learning trains a model on **labeled examples** — pairs of (input, correct answer). The model learns a function $f$ such that:

$$f(x) \approx y$$

where $x$ is an input image and $y \in \{\text{Dog}, \text{Cat}\}$.

---

## 1. The Big Picture

```mermaid
flowchart TD
    A[("Labeled Dataset- 🐶 dog_001.jpg → Dog- 🐱 cat_002.jpg → Cat- ...etc")]
    A --> B[Split Data]
    B --> C[Training Set- ~80%]
    B --> D[Test Set- ~20%]
    C --> E[Train Model]
    E --> F{Evaluate on- Test Set}
    F -->|Accuracy too low| E
    F -->|Accuracy acceptable| G[Deploy Model]
    G --> H["Predict on new- unlabeled image"]
```

---

## 2. Building the Training Dataset

Each example in the dataset is a **(feature, label)** pair. For images, features are pixel values.

```mermaid
flowchart LR
    subgraph Raw["Raw Images (unlabeled)"]
        I1[photo_01.jpg]
        I2[photo_02.jpg]
        I3[photo_03.jpg]
    end

    subgraph Labeling["Human Annotation"]
        L[Annotator assigns- ground-truth label]
    end

    subgraph Dataset["Labeled Dataset"]
        D1["(photo_01.jpg, Dog)"]
        D2["(photo_02.jpg, Cat)"]
        D3["(photo_03.jpg, Dog)"]
    end

    I1 --> L --> D1
    I2 --> L --> D2
    I3 --> L --> D3
```

> **Key point:** The model never sees the labels during inference — they exist only to guide training.

---

## 3. The Training Loop

Training iterates over the dataset repeatedly (each full pass = one **epoch**).

```mermaid
sequenceDiagram
    participant DS as Dataset
    participant M  as Model
    participant LF as Loss Function
    participant OP as Optimizer

    loop Each Epoch
        loop Each Mini-Batch
            DS ->> M: Forward pass: image pixels
            M  ->> LF: Predicted probabilities<br/>[P(Dog)=0.3, P(Cat)=0.7]
            LF ->> LF: Compare prediction vs true label<br/>loss = CrossEntropy(ŷ, y)
            LF ->> OP: Compute gradients ∂loss/∂weights
            OP ->> M: Update weights (gradient descent)
        end
        M -->> DS: Epoch complete — shuffle data
    end
```

**Loss decreases** as the model improves. Training stops when loss plateaus or validation accuracy peaks.

---

## 4. Decision Boundary

The model learns to separate the feature space into regions:

```mermaid
quadrantChart
    title Feature Space: Ear Shape vs Snout Length
    x-axis Short Snout --> Long Snout
    y-axis Rounded Ears --> Pointed Ears
    quadrant-1 Likely Cat
    quadrant-2 Likely Cat
    quadrant-3 Likely Dog
    quadrant-4 Likely Dog
    Dog A: [0.7, 0.2]
    Dog B: [0.8, 0.3]
    Dog C: [0.6, 0.15]
    Cat A: [0.2, 0.8]
    Cat B: [0.3, 0.75]
    Cat C: [0.15, 0.85]
    New Image: [0.25, 0.7]
```

The **decision boundary** is the line the model draws between classes. Points on one side → Dog; the other → Cat.

---

## 5. Inference Pipeline

Once trained, the model classifies new images it has never seen:

```mermaid
flowchart LR
    A["New Image- (unlabeled)"] --> B["Preprocessing- Resize · Normalize"]
    B --> C["Trained Model- (frozen weights)"]
    C --> D["Output Layer- Softmax"]
    D --> E["P(Dog) = 0.92- P(Cat) = 0.08"]
    E --> F["Predicted Class:- 🐶 Dog"]
```

The **softmax** function converts raw scores (logits) into probabilities that sum to 1:

$$P(\text{Dog}) = \frac{e^{z_{\text{Dog}}}}{e^{z_{\text{Dog}}} + e^{z_{\text{Cat}}}}$$

---

## 6. Evaluating the Model

A **confusion matrix** shows where the model succeeds and fails:

```mermaid
block-beta
    columns 3
    space:1 PredDog["Predicted: Dog"] PredCat["Predicted: Cat"]
    ActDog["Actual: Dog"]:1 TP["✅ True Positive- (correctly called Dog)"] FN["❌ False Negative- (missed a Dog)"]
    ActCat["Actual: Cat"]:1 FP["❌ False Positive- (wrongly called Dog)"] TN["✅ True Negative- (correctly called Cat)"]
```

Key metrics derived from the confusion matrix:

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | $(TP + TN) / N$ | Overall correct predictions |
| **Precision** | $TP / (TP + FP)$ | Of predicted Dogs, how many were Dogs? |
| **Recall** | $TP / (TP + FN)$ | Of actual Dogs, how many did we catch? |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean of precision & recall |

---

## 7. Supervised vs Unsupervised (Quick Contrast)

```mermaid
flowchart LR
    subgraph Supervised
        S1["Labeled images"] --> S2["Model learns:- Dog features vs Cat features"]
        S2 --> S3["Classify new image- as Dog or Cat"]
    end

    subgraph Unsupervised
        U1["Unlabeled images"] --> U2["Model discovers:- 2 clusters of similar images"]
        U2 --> U3["Cluster A and Cluster B- (no names assigned)"]
    end
```

Supervised learning **requires labels** but produces interpretable, task-specific outputs. Unsupervised learning finds hidden structure but doesn't know what the clusters *mean*.

---

## Key Vocabulary

| Term | Definition |
|---|---|
| **Label** | The correct answer for a training example (`Dog` or `Cat`) |
| **Feature** | An input variable used for prediction (pixel values, edge intensities) |
| **Epoch** | One full pass through the training dataset |
| **Loss** | Numeric measure of how wrong the model's prediction was |
| **Gradient Descent** | Optimization algorithm that iteratively reduces loss |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Generalization** | Model performs well on unseen examples |

