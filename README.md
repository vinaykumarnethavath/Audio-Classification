# 🎧 Environmental Sound Classification with Transformer Models

This project fine-tunes **Wav2Vec2**, **HuBERT**, and **AST (Audio Spectrogram Transformer)** on the **UrbanSound8K** dataset to classify environmental sounds like sirens, dog barks, drilling, etc.
It also implements ensemble techniques to boost accuracy.

## 🚀 Features
- Fine-tunes **Transformer-based audio models** (Wav2Vec2, HuBERT, AST)
- Performs **zero-shot**, **fine-tuning**, and **ensemble** evaluations
- Achieved **~94% accuracy** on UrbanSound8K
- Fully compatible with **Google Colab**

## 🧰 Tech Stack
Python, PyTorch, Hugging Face Transformers, Torchaudio, Librosa, Scikit-learn

## 📦 Setup
```bash
pip install -r requirements.txt
```

## 🗂 Dataset Setup

Download **UrbanSound8K** dataset from:  
🔗 https://urbansounddataset.weebly.com/urbansound8k.html  

After downloading, create this folder structure **(do not upload the dataset to GitHub)**:

```
UrbanSound-Transformers-Project/
│
├── src/
│   └── main.ipynb
│
├── UrbanSound8K/
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   ├── ...
│   ├── fold10/
│   └── UrbanSound8K.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

If using **Google Colab**, you can also mount Drive and access it from there:
```python
from google.colab import drive
drive.mount('/content/drive')
BASE_PATH = "/content/drive/MyDrive/UrbanSound8K"
```

## 🧠 Training
Run the notebook:
```
src/main.ipynb
```

## 🏁 Results
| Model | Accuracy | Precision | Recall | F1 |
|-------|-----------|------------|--------|----|
| Wav2Vec2-AM | 93% | 0.93 | 0.93 | 0.93 |
| HuBERT-LM | 92% | 0.92 | 0.92 | 0.92 |
| AST | 91% | 0.91 | 0.91 | 0.91 |
| Ensemble (AM+LM+AST) | **94%** | **0.94** | **0.94** | **0.94** |

## 📜 License
MIT
