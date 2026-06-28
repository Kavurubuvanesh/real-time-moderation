import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class JigsawDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize with DistilBERT's specific requirements
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            # Multi-label classification requires float tensors for BCEWithLogitsLoss
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def get_data_loaders(batch_size=32, max_length=128):
    print("Pulling Jigsaw dataset from Hugging Face...")
    # Programmatic pull bypasses manual Kaggle CSV downloads
    dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")

    # Convert to pandas for vectorized slicing
    train_df = pd.DataFrame(dataset['train'])

    # The 6 target classes for multi-label classification
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    texts = train_df['comment_text'].values
    labels = train_df[label_cols].values

    print("Splitting data into train and validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("Initializing DistilBertTokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = JigsawDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = JigsawDataset(val_texts, val_labels, tokenizer, max_length)

    # Pin memory to True for faster CPU-to-GPU data transfer during training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")
    return train_loader, val_loader, tokenizer