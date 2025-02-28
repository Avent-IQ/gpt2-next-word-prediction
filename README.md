# Next Word Prediction With GPT2

## 📌 Overview

This repository hosts the quantized version of the GPT2 model fine-tuned for next word prediction tasks. The model has been trained on the bookcorpus dataset from Hugging Face. The model is quantized to Float16 (FP16) to optimize inference speed and efficiency while maintaining high performance.

## 🏗 Model Details

- **Model Architecture:** GPT2
- **Task:** Next Word Prediction  
- **Dataset:** Hugging Face's `bookcorpus`  
- **Quantization:** Float16 (FP16) for optimized inference  
- **Fine-tuning Framework:** Hugging Face Transformers  

## 🚀 Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/gpt2-next-word-prediction"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```

### Question Answer Example

```python
# Input text
text = "Hi! How are"

# Tokenize input text
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

# Generate next word (max_length ensures we get only the next token)
output = model.generate(input_ids, max_length=input_ids.shape[1] + 1, do_sample=False)

# Decode output
generated_text = tokenizer.decode(output[0])

print("Generated Sentence:", generated_text)
```

## ⚡ Quantization Details

Post-training quantization was applied using PyTorch's built-in quantization framework. The model was quantized to Float16 (FP16) to reduce model size and improve inference efficiency while balancing accuracy.

## Evaluation Metrics

A well-trained language model should have a perplexity closer to 10–50, depending on the dataset and domain and our model's perplexity score is 32.4.

## 🔧 Fine-Tuning Details

### Dataset
The **Bookcorpus** dataset was used for training and evaluation. The dataset consists of **texts**.

### Training Configuration
- **Number of epochs**: 3
- **Batch size**: 8  
- **Learning rate**: 5e-5  
- **Evaluation strategy**: steps


## 📂 Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## ⚠️ Limitations

- The model may struggle for out of scope tasks.
- Quantization may lead to slight degradation in accuracy compared to full-precision models.
- Performance may vary across different writing styles and sentence structures.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
