from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from django.conf import settings


def load_datasets() -> Tuple[List[str], List[int]]:
    raw_dir = Path(settings.RAW_DATA_DIR)
    frames = []

    # News_dataset/Fake.csv & True.csv - handle both with and without space
    for news_dir_name in ['News_dataset', 'News _dataset']:
        news_dir = raw_dir / news_dir_name
        fake_path = news_dir / 'Fake.csv'
        true_path = news_dir / 'True.csv'
        if fake_path.exists() and true_path.exists():
            df_fake = pd.read_csv(fake_path)
            df_true = pd.read_csv(true_path)
            def build_text(df: pd.DataFrame) -> pd.Series:
                title = df['title'] if 'title' in df.columns else ''
                body = df['text'] if 'text' in df.columns else ''
                if isinstance(title, pd.Series) and isinstance(body, pd.Series):
                    return (title.fillna('') + ' ' + body.fillna('')).str.strip()
                if isinstance(body, pd.Series):
                    return body.fillna('').astype(str)
                return pd.Series([], dtype=str)
            frames.append(pd.DataFrame({'text': build_text(df_fake), 'label': 'fake'}))
            frames.append(pd.DataFrame({'text': build_text(df_true), 'label': 'real'}))
            break  # Found and loaded

    # data/raw/Fake.csv & True.csv
    fake_path2 = raw_dir / 'Fake.csv'
    true_path2 = raw_dir / 'True.csv'
    if fake_path2.exists() and true_path2.exists():
        df_fake = pd.read_csv(fake_path2)
        df_true = pd.read_csv(true_path2)
        def build_text2(df: pd.DataFrame) -> pd.Series:
            title = df['title'] if 'title' in df.columns else ''
            body = df['text'] if 'text' in df.columns else ''
            if isinstance(title, pd.Series) and isinstance(body, pd.Series):
                return (title.fillna('') + ' ' + body.fillna('')).str.strip()
            if isinstance(body, pd.Series):
                return body.fillna('').astype(str)
            return pd.Series([], dtype=str)
        frames.append(pd.DataFrame({'text': build_text2(df_fake), 'label': 'fake'}))
        frames.append(pd.DataFrame({'text': build_text2(df_true), 'label': 'real'}))

    # Generic support
    for name in ['kaggle_fake_news.csv', 'liar_dataset.csv']:
        path = raw_dir / name
        if path.exists():
            df = pd.read_csv(path)
            text_col = 'text' if 'text' in df.columns else ('statement' if 'statement' in df.columns else None)
            label_col = 'label' if 'label' in df.columns else ('verdict' if 'verdict' in df.columns else None)
            if text_col and label_col:
                frames.append(df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'}))
    if not frames:
        raise FileNotFoundError('No datasets found. Supported: data/raw/News_dataset/Fake.csv & True.csv, or data/raw/Fake.csv & True.csv, or kaggle_fake_news.csv/liar_dataset.csv.')
    df = pd.concat(frames, ignore_index=True).dropna()
    df['label'] = df['label'].astype(str).str.lower().replace({'false': 'fake', 'true': 'real'})
    df = df[df['label'].isin(['fake', 'real'])]
    y = (df['label'] == 'fake').astype(int).tolist()
    X = df['text'].tolist()
    return X, y


def run(model_name: str = 'distilbert-base-uncased', epochs: int = 1, batch_size: int = 32, max_len: int = 128, max_samples: int | None = 500):
    # Lazy import to avoid importing TF-Transformers stack at module import time
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    
    # Ensure max_samples defaults to 500 if None or 0
    if max_samples is None or max_samples <= 0:
        max_samples = 500
    
    print(f'Loading datasets (max {max_samples} samples)...')
    texts, labels = load_datasets()
    
    # Limit dataset size to max_samples
    if len(texts) > max_samples:
        print(f'Limiting dataset from {len(texts)} to {max_samples} samples for faster training')
        indices = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    else:
        print(f'Using all {len(texts)} samples (less than max_samples={max_samples})')
    
    split = int(0.8 * len(texts))
    train_texts, test_texts = texts[:split], texts[split:]
    y_train, y_test = np.array(labels[:split]), np.array(labels[split:])
    
    print(f'Training on {len(train_texts)} samples, testing on {len(test_texts)} samples')
    print(f'Using model: {model_name}, batch_size: {batch_size}, max_len: {max_len}, epochs: {epochs}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('Tokenizing data...')
    def tokenize(ds_texts):
        return tokenizer(ds_texts, truncation=True, padding=True, max_length=max_len, return_tensors='tf')

    train_enc = tokenize(train_texts)
    test_enc = tokenize(test_texts)

    print('Loading model...')
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        from_pt=False  # force TF weights; do not attempt PT->TF conversion
    )

    # Optimizer compatible across TF versions (use legacy if available)
    try:
        AdamOpt = tf.keras.optimizers.legacy.Adam  # type: ignore[attr-defined]
    except Exception:
        AdamOpt = tf.keras.optimizers.Adam
    optimizer = AdamOpt(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.metrics.SparseCategoricalAccuracy('accuracy')]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print('Starting training...')
    # Use smaller validation split and limit validation steps
    validation_steps = max(1, len(test_texts) // (batch_size * 4))  # Limit validation steps
    model.fit(
        dict(train_enc), y_train, 
        validation_data=(dict(test_enc), y_test),
        validation_steps=validation_steps,
        epochs=epochs, 
        batch_size=batch_size,
        verbose=1
    )
    
    print('Evaluating...')
    eval_out = model.evaluate(dict(test_enc), y_test, verbose=1, batch_size=batch_size)
    print('BERT eval:', eval_out)

    models_dir = Path(settings.MODELS_DIR) / 'bert'
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f'Saving model to {models_dir}...')
    model.save_pretrained(models_dir.as_posix())
    tokenizer.save_pretrained(models_dir.as_posix())
    print('BERT training completed!')


if __name__ == '__main__':
    run()


