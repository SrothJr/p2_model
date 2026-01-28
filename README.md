# Bangla Depression Detection Model

This repository contains a fine-tuned transformer model for Bangla text depression detection.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SrothJr/p2_model.git
   cd p2_model
   ```

2. **Create a virtual environment:**

   ```bash
   # On Windows
   python -m venv venv

   # On macOS/Linux
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**

   ```bash
   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the notebook:**
   ```bash
   jupyter notebook training.ipynb
   ```

## Project Structure

- `training.ipynb` - Main training and evaluation notebook
- `dataset.xlsx` - Dataset file with posts and labels
- `requirements.txt` - Python dependencies

## Model Information

The model is a fine-tuned transformer-based classifier for Bangla depression detection with 4 classes:

- Class 0: Normal (label 1)
- Class 1: Mild (label 2)
- Class 2: Moderate (label 3)
- Class 3: Severe (label 4)

### Using the Pre-trained Model (No Retraining Needed)

The pre-trained model is available on the Hugging Face Model Hub. You have two options:

#### Option 1: Direct Download & Inference (Quickest)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "SrothJr/bangla-depression-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Inference example
text = "আপনার বাংলা টেক্সট এখানে"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class = logits.argmax(-1).item()
print(f"Predicted depression level: {predicted_class}")
# 0: Normal, 1: Mild, 2: Moderate, 3: Severe
```

#### Option 2: Clone Repo & Setup Environment

If you want to run the full training notebook or develop locally:

```bash
# Follow the "Getting Started" section above first, then:
pip install -r requirements.txt

# The model will be automatically downloaded when you run:
jupyter notebook training.ipynb
```

#### Model Card

View the full model card and documentation here:
👉 **[SrothJr/bangla-depression-model on Hugging Face](https://huggingface.co/SrothJr/bangla-depression-model)**

