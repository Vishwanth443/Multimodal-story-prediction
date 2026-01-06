# Multimodal-story-prediction
Deep Neural Networks and Learning Systems Final Project - Multimodal Story Continuation
## Module
**Deep Neural Networks and Learning Systems (55-710365)**  
Level 7 – MSc  

## Assignment Title
**Multimodal Sequence Modelling**

## Author
**Individual Coursework Submission**

---

## 1. Project Overview

This project addresses the task of **multimodal visual story reasoning**, where a model is required to understand and reason over sequences of images and associated textual descriptions in order to predict the next narrative action in a story.

The work is based on the **StoryReasoning Dataset**, which contains sequential image–text pairs representing evolving visual narratives. The project implements and evaluates:

- A baseline multimodal architecture using feature concatenation  
- A proposed architecture using cross-modal attention for improved multimodal alignment  

The objective is to investigate whether explicit cross-modal interactions between visual and textual features improve story understanding compared to simple fusion strategies.

---

## 2. Problem Statement

Baseline multimodal architectures often rely on direct concatenation of image and text features, which provides only shallow interaction between modalities. Such approaches struggle to align specific visual regions with corresponding textual cues, leading to weak multimodal grounding.

In the context of visual storytelling, this limitation is critical, as narrative understanding depends on accurately linking objects, actions, and scenes across time.

This project proposes a **cross-modal attention mechanism** to address this limitation and empirically evaluates its effectiveness.

---

## 3. Dataset Description

The project uses the **StoryReasoning Dataset**, which includes:

- Sequential images representing story frames  
- Associated textual descriptions  
- Temporal ordering of frames  

### Action Labels

Each story description is mapped to an action category extracted from the text:


## 4. Repository Structure
multimodal-story-reasoning/
│
├── experiments.py # Main experiment pipeline
├── config.yaml # Hyperparameters and settings
├── README.md # Project documentation
│
├── src/
│ ├── dataset.py # Dataset wrapper and label extraction
│ ├── visual_encoder.py # CNN-based image encoder (ResNet-18)
│ ├── text_encoder.py # Transformer-based text encoder (BERT)
│ ├── model_baseline.py # Baseline concatenation model
│ ├── model_proposed.py # Proposed cross-modal attention model
│ ├── train.py # Training loop with logging
│ ├── evaluate.py # Accuracy and confusion matrix generation
│ └── utils.py # Plotting utilities
│
├── results/
│ ├── baseline_loss.csv
│ ├── proposed_loss.csv
│ ├── baseline_loss.png
│ ├── proposed_loss.png
│ ├── baseline_confusion.png
│ ├── proposed_confusion.png
│ └── comparison.csv




---

## 5. Model Architectures

### 5.1 Baseline Model – Concatenation Fusion

- **Visual Encoder:** ResNet-18 (pretrained, frozen)  
- **Text Encoder:** BERT-base (pretrained, frozen)  
- **Fusion Strategy:** Feature concatenation  
- **Classifier:** Fully connected layers  

This model serves as the control condition for evaluation.

### 5.2 Proposed Model – Cross-Modal Attention

- Uses the same visual and text encoders as the baseline  
- Introduces a multi-head cross-modal attention layer  
- Text embeddings attend over visual features  
- Produces a fused representation with stronger multimodal alignment  

This architecture is inspired by modern vision-language models such as **ViLBERT** and **BLIP**.

---

## 6. Training Procedure

Training is performed using supervised classification with the following setup:

- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  
- **Device:** CPU  
- **Batch Size:** 1  
- **Epochs:** 3  
- **Frames per Story:** 2  

During training:

- Batch-level loss is printed in real time  
- Loss values are saved to CSV files  
- Average loss per epoch is reported  

Example training output:

[INFO] Epoch 2
[TRAIN] Batch 10 | Loss 1.9255
[EPOCH DONE] Avg Loss: 1.6401


---

## 7. Execution Instructions

### 7.1 Install Dependencies

Ensure the following libraries are installed:

- Python 3.9 or later  
- PyTorch  
- Torchvision  
- Transformers  
- Scikit-learn  
- Pandas  
- Matplotlib  
- Seaborn  
- PyYAML  

### 7.2 Run the Experiment

From the project root directory:

python experiments.py



This command executes the complete pipeline:

- Load configuration  
- Load and preview dataset  
- Train baseline model  
- Evaluate baseline model  
- Train proposed model  
- Evaluate proposed model  
- Generate plots and comparison tables  

---

## 8. Evaluation Metrics

### 8.1 Accuracy

Overall classification accuracy is computed for both models.

### 8.2 Confusion Matrix

Confusion matrices are generated and saved as images:

- `baseline_confusion.png`  
- `proposed_confusion.png`  

These visualise class-wise prediction behaviour.

### 8.3 Training Loss Curves

Loss curves are plotted from CSV logs:

- `baseline_loss.png`  
- `proposed_loss.png`  

---

## 9. Results and Comparison


Baseline Accuracy : 0.3111
Proposed Accuracy : 0.3049


### Comparison Table

Saved as:

results/comparison.csv


| Model     | Validation Accuracy |
|-----------|---------------------|
| Baseline  | 0.3111              |
| Proposed  | 0.3049              |

---

## 10. Analysis and Discussion

The experimental results show that the proposed cross-modal attention model does not significantly outperform the baseline under the current training constraints. This outcome is attributed to:

- Limited training epochs  
- Small batch size  
- Frozen visual and text encoders  
- CPU-only execution  

Despite this, the implementation demonstrates:

- Correct multimodal data handling  
- Functional cross-modal attention mechanism  
- Fully reproducible experimental evaluation  

---

## 11. Future Work

Potential future improvements include:

- Training for additional epochs  
- Fine-tuning BERT and CNN encoders  
- Increasing batch size  
- Introducing multimodal contrastive losses  
- GPU acceleration  
- Extending the task to text and image generation  

---

## 12. Conclusion

This project presents a complete, modular, and reproducible multimodal learning pipeline for visual story reasoning. It compares a baseline fusion strategy with a more advanced cross-modal attention approach and provides a strong foundation for future research in multimodal sequence modelling.

---

## 13. Dataset Reference

Oliveira, D. A. P., & Matos, D. M. (2025).  
*StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation.*  
arXiv:2505.10292
