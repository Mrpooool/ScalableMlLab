# **Lab Assignment 1** Building and Evaluating Convolutional Neural Networks

You can use Lab 2 notebook Part B to implement this assignment.

In this lab, your task is to use a practical ML workflow to build image classifiers with AlexNet and ResNet, investigate how design choices affect learning, and compare accuracy, robustness, and computational cost.

Use CIFAR-10, which contains labelled colour images from ten classes. The dataset is available from the official website and can be loaded with torchvision.

| Dataset | Characteristics | Link |
| :---- | :---- | :---- |
| CIFAR-10 | RGB images in ten classes | [Link](https://www.cs.toronto.edu/~kriz/cifar.html) |

At the end of this lab, your notebook should:

* prepare the image data for reproducible training and evaluation,  
* train and compare AlexNet and ResNet,  
* investigate model capacity and regularization,  
* evaluate performance on clean, corrupted, and adversarial inputs,  
* support a recommendation using measured accuracy, robustness, and cost.

## Task 1\. Study and Prepare the Data

Study CIFAR-10 before building the models. Create training and validation partitions from the provided training data, and reserve the test data for final evaluation. Prepare reusable data loading and preprocessing steps.

Start from and extend the following questions:

* What does an example represent, and how are the classes distributed?  
* Which preprocessing steps are needed for the two architectures?  
* How is the dataset preprocessed and splitted?  
* How to control the randomness in the experiment?

## Task 2\. Implement and Compare the Models

Implement AlexNet and ResNet using PyTorch. Use the recommended CIFAR-10 adaptations described in [Part A9](https://colab.research.google.com/drive/1ZsygMpALqkvHszLP_k1WaR8bilzjE3Iz?usp=sharing) as a starting point. Explain the changes from the original architectures and justify any further modifications. Train both models using a reproducible procedure and compare them under a clearly stated compute budget.

Describe and justify:

* the layers, feature-map dimensions, and parameter counts,  
* the input preparation and classification head,  
* the convolution operations, and the roles they play in the visual task,  
* the optimization and checkpoint-selection procedure,  
* the role of residual connections in ResNet.

You may explore other variants after establishing the two baseline models. Explain any changes so that the reader can distinguish an original architecture from an experimental variant.

## Task 3\. Investigate Capacity and Regularization

Explore how model capacity and regularization affect learning and generalization. Compare a small set of different widths or depths, and investigate L2 regularization and dropout. Include a condition without dropout and compare it with two dropout settings of your choice.

For each comparison:

* state what is changed and what is held fixed,  
* show training and validation behaviour,  
* explain evidence of underfitting, overfitting, or optimization difficulty,  
* discuss the effect on accuracy and computational cost.

Choose settings that are feasible for your available computational resource. Support your interpretation with controlled comparisons rather than a large, unexplained parameter search.

## Task 4\. Evaluate Robustness

Investigate whether training with data augmentation improves robustness. Choose transformations that are appropriate for the images, and construct a reproducible corrupted evaluation set, for example using rotation, geometric distortion, or noise.

Compare the original and augmented training conditions in terms of:

* performance on clean images,  
* performance on corrupted images,  
* the types of errors introduced by different transformations,  
* the additional training or evaluation cost.

Use the same evaluation images and corruption settings for the models being compared. Explain your choices and discuss whether stronger transformations can make labels ambiguous.

## Task 5\. Analyse Predictions and Adversarial Examples

Evaluate the models quantitatively and inspect representative successful and unsuccessful predictions. Use suitable metrics and visualizations to explain which classes or image characteristics are difficult.

Your analysis should include:

* a numerical comparison and training diagnostics,  
* examples of correct and incorrect predictions,  
* an adversarial example that you generate against a trained model,  
* the perturbation constraint, attack method, and observed effect.

Distinguish sensitivity to random corruption from a deliberately optimized perturbation. If your attempted attack does not change a prediction within the chosen constraint, report that result and explain the limitations of the attempt.

Task 6\. Benchmark and Discuss the Tradeoffs

Compare the configurations under the compute budget you defined. Measure training cost and inference performance, and relate them to accuracy and robustness.

Discuss:

* Which model would you choose under your available computational budget, and why?  
* Which changes improve accuracy or robustness at an acceptable cost?  
* Which conclusions are sensitive to the training duration, random seed, or hardware?  
* What further experiment would provide the most useful evidence?

Support your conclusions with measurements from your implementation. 

## Challenge Optional Extensions

These challenges are optional. Attempt either or both after the main tasks. Choose and justify a feasible method and evaluation protocol. Include your implementation, evidence and discussion in Part B.

### **Challenge 1 Model Compression**

Explore a method to make a trained model smaller or faster while retaining acceptable predictive performance. Define an accuracy tolerance and efficiency measure, and compare with the uncompressed reference.

* Describe the method and any additional training or fine-tuning.  
* Measure accuracy and relevant costs consistently. Distinguish parameter or file-size reduction from actual inference speed.  
* Discuss the tradeoff and account for the additional compression compute.

### **Challenge 2 Generalization**

Evaluate a selected model on an unseen corruption family or a different, suitably labelled image domain. Keep these conditions separate from model and augmentation selection.

* Explain the shift, label compatibility and preprocessing.  
* Use held-out evaluation and paired image identities when applicable.  
* Discuss transfer, failure cases, severity and limits of generalization.

## Deliverables

Submit **one executed notebook** containing:

* the complete implementation, configuration, and reproduction instructions,  
* the experimental results, figures, and numerical comparisons,  
* the dataset, architecture, documentation, and code references used,

and **a report** containing discussions and conclusions for Tasks 1-6 and challenges attempted. The notebook should be executed so it shows the final results you are using in the report. If the Colab limit is restricted to run all tasks, you may submit separate notebooks addressing different tasks, but you need to discuss all tasks in a single report.