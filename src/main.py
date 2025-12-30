
!pip install transformers datasets torchaudio librosa evaluate jiwer scikit-learn timm -q

from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive/urbansound8k"
METADATA_PATH = BASE_PATH + "/UrbanSound8K.csv"
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(METADATA_PATH)

print("--- Dataset Head ---")
print(df.head())
print("\n--- Classes ---")
print(df['class'].unique())
print("\n--- Class Distribution ---")
print(df['class'].value_counts())
df['class'].value_counts().plot(kind='bar', figsize=(10,5), title="Class Distribution")
plt.show()
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class UrbanSoundDataset(Dataset):
    def __init__(self, df, base_path, processor, target_sr=16000):
        self.df = df
        self.base_path = base_path
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.base_path, f"fold{row['fold']}", row["slice_file_name"])

        waveform, sr = torchaudio.load(filepath)
        waveform = waveform.squeeze()
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
        inputs = self.processor(waveform, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs["labels"] = torch.tensor(row["classID"], dtype=torch.long)
        return inputs

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['classID'], random_state=42)

import librosa
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sklearn.metrics import classification_report

class UrbanSoundDataset(Dataset):
    def __init__(self, df, base_path, processor, max_length=16000*4):
        self.df = df
        self.base_path = base_path
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.base_path, f"fold{row['fold']}", row['slice_file_name'])
        speech_array, sr = librosa.load(file_path, sr=16000)

        if len(speech_array) > self.max_length:
            speech_array = speech_array[:self.max_length]
        else:
            pad_len = self.max_length - len(speech_array)
            speech_array = np.pad(speech_array, (0, pad_len), mode="constant")

        inputs = self.processor(speech_array, sampling_rate=16000, return_tensors="pt", padding="longest")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(row["classID"], dtype=torch.long)
        return item

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
zero_shot_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(df['class'].unique())
).to("cuda")


test_dataset = UrbanSoundDataset(test_df, BASE_PATH, processor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
from torch.cuda.amp import autocast

zero_shot_model.eval()
y_pred_zero_shot, y_true = [], []

with torch.no_grad():
    for batch in test_loader:
        inputs = {k: v.to("cuda", non_blocking=True) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to("cuda", non_blocking=True)
        with autocast():
            outputs = zero_shot_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

        y_pred_zero_shot.extend(preds.tolist())
        y_true.extend(labels.tolist())
print("\n--- Zero-Shot Wav2Vec2 ---")
print(classification_report(y_true, y_pred_zero_shot, target_names=df['class'].unique()))

#  Fine-Tune Wav2Vec2 Acoustic Model (AM)

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence

num_epochs = 5
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
output_dir = "./wav2vec2-am"

def collate_fn(batch):
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    input_values = [item["input_values"].view(-1) for item in batch]
    input_values_padded = pad_sequence(input_values, batch_first=True)

    batch_out = {"input_values": input_values_padded, "labels": labels}

    if "attention_mask" in batch[0]:
        att_masks = [item["attention_mask"].view(-1) for item in batch]
        att_masks_padded = pad_sequence(att_masks, batch_first=True)
        batch_out["attention_mask"] = att_masks_padded

    return batch_out

train_dataset = UrbanSoundDataset(train_df, BASE_PATH, processor)
eval_dataset = UrbanSoundDataset(test_df, BASE_PATH, processor)

am_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(df['class'].unique())
).to("cuda")

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=1, 
    logging_dir="./logs",
    logging_steps=100,
)
trainer = Trainer(
    model=am_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=None,
    data_collator=collate_fn,
)
for epoch in range(num_epochs):
    print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========")

    trainer.args.num_train_epochs = 1
    trainer.train(resume_from_checkpoint=None)

    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}")
    trainer.save_model(ckpt_dir)
    print(f"Checkpoint saved at {ckpt_dir}")

    preds_out = trainer.predict(eval_dataset)
    logits = preds_out.predictions
    y_true = preds_out.label_ids
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print(f"Eval results — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

print("\n✅ Training complete. Final model saved to:", output_dir)
from transformers import HubertForSequenceClassification

lm_model = HubertForSequenceClassification.from_pretrained(
    "facebook/hubert-base-ls960", num_labels=len(df['class'].unique())
).to("cuda")

training_args.output_dir = "./hubert-lm"

trainer_lm = Trainer(
    model=lm_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,
    data_collator=default_data_collator,
)

trainer_lm.train()

from transformers import ASTForAudioClassification, ASTFeatureExtractor
feature_extractor_ast = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(df['class'].unique()),
    ignore_mismatched_sizes=True
).to("cuda")

train_dataset_ast = UrbanSoundDataset(train_df, BASE_PATH, feature_extractor_ast)
test_dataset_ast = UrbanSoundDataset(test_df, BASE_PATH, feature_extractor_ast)

training_args.output_dir = "./ast"

trainer_ast = Trainer(
    model=ast_model,
    args=training_args,
    train_dataset=train_dataset_ast,
    eval_dataset=test_dataset_ast,
    tokenizer=feature_extractor_ast,
    data_collator=default_data_collator,
)

num_epochs = 5
output_dir = "./ast"

for epoch in range(num_epochs):
    print(f"\n========== AST Epoch {epoch+1}/{num_epochs} ==========")

    trainer_ast.args.num_train_epochs = 1
    trainer_ast.train(resume_from_checkpoint=None)

    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}")
    trainer_ast.save_model(ckpt_dir)
    print(f"Checkpoint saved at {ckpt_dir}")

    # Evaluate
    preds_out = trainer_ast.predict(test_dataset_ast)
    logits = preds_out.predictions
    y_true = preds_out.label_ids
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print(f"Eval results — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

print("\n✅ AST training complete. Final model saved to:", output_dir)
preds_am = trainer_am.predict(test_dataset)
y_pred_am = np.argmax(preds_am.predictions, axis=1)

preds_lm = trainer_lm.predict(test_dataset)
y_pred_lm = np.argmax(preds_lm.predictions, axis=1)

preds_ast = trainer_ast.predict(test_dataset_ast)
y_pred_ast = np.argmax(preds_ast.predictions, axis=1)

logits_am = preds_am.predictions
logits_lm = preds_lm.predictions
logits_ast = preds_ast.predictions

ensemble_logits_full = (logits_am + logits_lm + logits_ast) / 3
y_pred_ensemble_full = np.argmax(ensemble_logits_full, axis=1)

ensemble_logits_am_lm = (logits_am + logits_lm) / 2
y_pred_ensemble_am_lm = np.argmax(ensemble_logits_am_lm, axis=1)


results_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

models_to_compare = [
    ("Zero-Shot Wav2Vec2", y_pred_zero_shot),
    ("Wav2Vec2-AM", y_pred_am),
    ("HuBERT-LM", y_pred_lm),
    ("Ensemble (AM+LM)", y_pred_ensemble_am_lm),
    ("AST", y_pred_ast),
    ("Ensemble (AM+LM+AST)", y_pred_ensemble_full)
]

for name, preds in models_to_compare:
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted")
    new_row = pd.DataFrame([{
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
    }])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

print("\n--- FINAL MODEL COMPARISON (Reordered) ---")
print(results_df)

complexity_data = [
    {'Model': 'Wav2Vec2-AM', 'Approx. Parameters': '95 Million', 'Architecture / Input': 'Transformer on Raw Waveform',
     'Key Concept': 'Learns via contrastive loss on masked audio.'},
    {'Model': 'HuBERT-LM', 'Approx. Parameters': '95 Million', 'Architecture / Input': 'Transformer on Raw Waveform',
     'Key Concept': "Predicts discrete 'word-like' audio units."},
    {'Model': 'AST', 'Approx. Parameters': '86 Million', 'Architecture / Input': 'Transformer on Spectrogram',
     'Key Concept': 'Treats audio as an image for recognition.'},
    {'Model': 'Ensemble (AM+LM)', 'Approx. Parameters': '95M + 95M', 'Architecture / Input': 'Combination / Averaged Logits',
     'Key Concept': 'Combines two models for higher accuracy at 2x inference cost.'},
    {'Model': 'Ensemble (AM+LM+AST)', 'Approx. Parameters': '95M + 95M + 86M', 'Architecture / Input': 'Combination / Averaged Logits',
     'Key Concept': 'Combines all three models for max accuracy at 3x inference cost.'}
]

complexity_df = pd.DataFrame(complexity_data)

print("\n--- ARCHITECTURAL & COMPUTATIONAL COMPLEXITY 🧠 ---")
print(complexity_df.to_string())
