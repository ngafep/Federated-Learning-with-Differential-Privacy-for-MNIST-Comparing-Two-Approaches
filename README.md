# <span style="color:darkcyan">Federated Learning with Differential Privacy for MNIST:<br>Comparing Two Approaches</span>


This Jupyter Notebook contains implementations of Federated Learning models for the MNIST dataset, showcasing two different approaches to compare their effectiveness in terms of classification performance and privacy mechanisms.

### *`Overview`*

Federated Learning allows for decentralized model training while maintaining data privacy. On the other hand, differential privacy is a technique used to protect individuals' privacy when their data is included in a dataset. It ensures that the output of data analysis or a query doesn't reveal whether any individual's data was included. This is achieved by adding controlled noise to the results, making it difficult to identify specific information about any individual while still allowing for accurate insights into the overall dataset. Essentially, differential privacy provides a mathematical guarantee that any single individual's data doesn't significantly affect the outcome, protecting their privacy even if an attacker has some background knowledge.
In this repository, we compare two distinct Federated Learning architectures and privacy mechanisms for MNIST digit classification.

### *`Contents`*

- *<span style="color:red">Approach A</span>* *(Simple Feedforward Network):*<br> Chosen for its simplicity, computational efficiency, and as a baseline model to test the basic Federated Learning setup.

  
- *<span style="color:blue">Approach B</span>* *(Convolutional Neural Network - CNN):*<br> Selected for its superior performance in image classification tasks, leveraging the spatial feature extraction capabilities of convolutional layers to improve the accuracy of the model on the MNIST dataset.

### *`Comparison`*

Both *<span style="color:red">Approach A</span>* and *<span style="color:blue">Approach B</span>* implement federated learning with differential privacy, using different frameworks and approaches.

The choice of architecture in each approach reflects different design goals and considerations for Federated Learning (FL) in terms of complexity, performance, and generalization to the MNIST digit classification task.

#### *<span style="color:red">Approach A:</span>* *Simple Feedforward Neural Network*

- **Architecture:**
  - **Layers:** `Flatten` -> `Dense` -> `Dense`
  - **Activation:** ReLU for the hidden layer, Softmax for the output

- **Rationale:**
  - **Simplicity and Efficiency:** A feedforward neural network is straightforward to implement and computationally less expensive compared to more complex models like CNNs. This makes it suitable for environments with limited computational resources, which is often a concern in Federated Learning scenarios where client devices may have varying levels of computational power.
  
  - **Baseline Performance:** The simple feedforward network serves as a baseline model to establish a basic performance level. Despite its simplicity, this architecture can achieve reasonably good accuracy on MNIST, which is a relatively simple dataset.
  
  - **Use Case:** This architecture is often chosen when the primary goal is to demonstrate the feasibility of Federated Learning or when the focus is on low-resource settings rather than maximizing accuracy.

#### *<span style="color:blue">Approach B:</span>* *Convolutional Neural Network (CNN)*

- **Architecture:**
  - **Layers:** `Conv2d` -> `Linear`
  - **Activation:** ReLU for the convolutional layer, Log-Softmax for the output

- **Rationale:**
  - **Higher Performance:** CNNs are specifically designed for tasks involving spatial hierarchies, such as image classification. The convolutional layers can capture local patterns (like edges, textures, etc.), which are crucial for accurately classifying images. As a result, CNNs generally outperform feedforward networks on image datasets like MNIST.
  
  - **Generalization:** CNNs tend to generalize better on image data due to their ability to learn and extract spatial features, making them more robust and less prone to overfitting compared to fully connected networks, especially on larger datasets.
  
  - **Log-Softmax Activation:** The use of Log-Softmax in the output layer is common in classification tasks, particularly when combined with the negative log-likelihood loss, which offers numerical stability and better handling of probabilities in classification tasks.
  
  - **Use Case:** This approach is selected when the goal is to maximize model accuracy and leverage the strengths of CNNs in image classification tasks. It is particularly useful in Federated Learning setups where the client devices have sufficient computational resources to handle the more complex computations involved in CNNs.

### *`Remark:`*
Contrary to the initial expectation based on architectural differences, the feedforward network in *<span style="color:red">Approach A</span>* outperforms the CNN in *<span style="color:blue">Approach B</span>* in this specific federated learning setup for MNIST classification. This highlights the importance of not just relying on model architecture but also carefully considering the overall setup, including hyperparameters, data distribution, and training dynamics in federated learning.

#### *`Differential Privacy`*

1. *<span style="color:red">Approach A</span>*
   - **Differential Privacy Implementation:** Custom implementation adding noise to model weights.
   - **Noise Addition:** Manual, using `np.random.normal` to add noise to weights.

2. *<span style="color:blue">Approach B</span>*
   - **Differential Privacy Implementation:** Utilizes `Opacus` PrivacyEngine to handle differential privacy.
   - **Noise Addition:** Automated through `PrivacyEngine` during training.

#### *`Data Loading and Preprocessing`*

1. *<span style="color:red">Approach A</span>*
   - **Data Source:** MNIST from Keras.
   - **Preprocessing:** Normalization to [0, 1] range, one-hot encoding of labels.
   - **Client Data Splitting:** Manual splitting using `np.array_split`.

2. *<span style="color:blue">Approach B</span>*
   - **Data Source:** MNIST from `torchvision`.
   - **Preprocessing:** Normalization to [0, 1] range, one-hot encoding of labels.
   - **Client Data Splitting:** Manual splitting using `np.array_split`, converted to `DataLoader` for PyTorch.

#### *`Training Process`*

1. *<span style="color:red">Approach A</span>*
   - **Training Method:** Each client trains a local model, and weights are updated in the global model.
   - **Noise Addition:** Noise is added to weights after local training.
   - **Aggregation:** Weights are averaged for each layer.

2. *<span style="color:blue">Approach B</span>*
   - **Training Method:** Each client trains a local model using PyTorch, and local models are aggregated.
   - **Privacy and Aggregation:** Local models use differential privacy provided by `Opacus`.
   - **Aggregation:** Weights are averaged for each parameter.

#### *`Results`*

1. *<span style="color:red">Approach A</span>*
   - **Final Loss:** 0.10016341507434845
   - **Final Accuracy:** 0.9782000184059143
   - **Estimated Privacy Budget (ε):** 3.3833789404644246

2. *<span style="color:blue">Approach B</span>*
   - **Final Loss:** 0.2199314683675766
   - **Final Accuracy:** 0.9725000262260437
   - **Estimated Privacy Budget (ε):** 0.0004166666666666668

#### *`Key Differences`*

- **Framework:** *<span style="color:red">Approach A</span>* uses TensorFlow/Keras, while *<span style="color:blue">Approach B</span>* uses PyTorch with `Opacus`.
- **Privacy Implementation:** *<span style="color:red">Approach A</span>* manually adds noise to weights, whereas *<span style="color:blue">Approach B</span>* uses `Opacus` to handle differential privacy.
- **Model Type:** *<span style="color:red">Approach A</span>* uses a simple feedforward network, while *<span style="color:blue">Approach B</span>* uses a CNN, which is generally more effective for image classification tasks.

### *`Conclusion Based on Performance and Accuracy`*

1. *<span style="color:red">Approach A:</span>*
   - Shows good performance with accuracy and loss values improving steadily over rounds.
   - The final metrics are satisfactory, with high accuracy and acceptable privacy estimates.

2. *<span style="color:blue">Approach B:</span>*
   - The final results show that the model achieves competitive accuracy, but with some fluctuation in performance due to differential privacy constraints.
   - The noise multiplier and privacy guarantees are more rigorously managed.


