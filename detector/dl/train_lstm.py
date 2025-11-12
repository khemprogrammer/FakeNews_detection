from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


def run(max_words: int = 30000, max_len: int = 300, embedding_dim: int = 128):
    texts, labels = load_datasets()
    split = int(0.8 * len(texts))
    train_texts, test_texts = texts[:split], texts[split:]
    y_train, y_test = np.array(labels[:split]), np.array(labels[split:])

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_test = tokenizer.texts_to_sequences(test_texts)
    x_train = keras.utils.pad_sequences(x_train, maxlen=max_len)
    x_test = keras.utils.pad_sequences(x_test, maxlen=max_len)

    model = keras.Sequential([
        layers.Embedding(max_words, embedding_dim, input_length=max_len),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=64)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print('LSTM accuracy:', float(acc))

    models_dir = Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / 'lstm_model.keras')
    # Save tokenizer and parameters for prediction
    import joblib
    joblib.dump({'tokenizer': tokenizer, 'max_words': max_words, 'max_len': max_len}, 
                models_dir / 'lstm_tokenizer.joblib')


if __name__ == '__main__':
    run()


