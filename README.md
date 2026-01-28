# Bangla Depression Detection Model

This repository contains a fine-tuned transformer model for Bangla text depression detection.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
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
- `bangla_depression_1/` - Pre-trained model files (model weights, tokenizer config)
- `bangla_depression_model/` - Training checkpoints from different epochs
- `requirements.txt` - Python dependencies

## Model Information

The model is a fine-tuned transformer-based classifier for Bangla depression detection with 4 classes:
- Class 0: Normal (label 1)
- Class 1: Mild (label 2)
- Class 2: Moderate (label 3)
- Class 3: Severe (label 4)

## Notes

- The `venv/` folder is not included in the repository (see `.gitignore`)
- Follow the setup instructions above to create your own virtual environment
- Large model folders are tracked in `.gitignore` - ensure they're in the repo or download them separately as needed
