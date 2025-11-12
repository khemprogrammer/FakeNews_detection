from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from django.conf import settings
from detector.nlp.preprocess import build_tfidf, transform_tfidf


def load_datasets() -> Tuple[List[str], List[str]]:
    # Detect and load datasets from supported formats in data/raw
    raw_dir = Path(settings.RAW_DATA_DIR)
    frames = []

    # 1) News_dataset with Fake.csv and True.csv (Kaggle) - handle both with and without space
    for news_dir_name in ['News_dataset', 'News _dataset']:
        news_dir = raw_dir / news_dir_name
        fake_path = news_dir / 'Fake.csv'
        true_path = news_dir / 'True.csv'
        if fake_path.exists() and true_path.exists():
            df_fake = pd.read_csv(fake_path)
            df_true = pd.read_csv(true_path)
            # Common Kaggle columns: title, text, subject, date
            def build_text(df: pd.DataFrame) -> pd.Series:
                title = df['title'] if 'title' in df.columns else ''
                body = df['text'] if 'text' in df.columns else ''
                if isinstance(title, pd.Series) and isinstance(body, pd.Series):
                    return (title.fillna('') + ' ' + body.fillna('')).str.strip()
                if isinstance(body, pd.Series):
                    return body.fillna('').astype(str)
                return pd.Series([], dtype=str)

            df_fake = pd.DataFrame({'text': build_text(df_fake), 'label': 'fake'})
            df_true = pd.DataFrame({'text': build_text(df_true), 'label': 'real'})
            frames.append(df_fake)
            frames.append(df_true)
            break  # Found and loaded, no need to check other folder names

    # 2) Or top-level Fake.csv / True.csv directly under data/raw
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

    # 3) Generic CSVs (previous logic)
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
    df_all = pd.concat(frames, ignore_index=True).dropna()
    df_all['label'] = df_all['label'].astype(str).str.lower().replace({'fake': 'fake', 'false': 'fake', 'true': 'real', 'real': 'real'})
    df_all = df_all[df_all['label'].isin(['fake', 'real'])]
    return df_all['text'].tolist(), df_all['label'].tolist()


def evaluate_and_print(y_true, y_pred, name: str):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print(name)
    print('accuracy', acc)
    print('precision', prec, 'recall', rec, 'f1', f1)
    print(classification_report(y_true, y_pred))
    return acc, prec, rec, f1


def run(max_samples: int | None = 1000):
    import numpy as np
    
    print(f'Loading datasets (max {max_samples or "all"} samples)...')
    texts, labels = load_datasets()
    
    # Limit dataset size for faster training
    if max_samples and len(texts) > max_samples:
        print(f'Limiting dataset from {len(texts)} to {max_samples} samples for faster training')
        indices = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    else:
        print(f'Using all {len(texts)} samples')
    
    split = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split], labels[:split]
    test_texts, test_labels = texts[split:], labels[split:]

    print(f'Training on {len(train_texts)} samples, testing on {len(test_texts)} samples')

    print('Building TF-IDF vectorizer...')
    vec_bundle, X_train = build_tfidf(train_texts)
    X_test = transform_tfidf(vec_bundle, test_texts)

    # NB
    print('Training Naive Bayes...')
    nb = MultinomialNB()
    nb.fit(X_train, train_labels)
    nb_pred = nb.predict(X_test)
    evaluate_and_print(test_labels, nb_pred, 'Naive Bayes')

    # RF (reduced n_estimators for faster training)
    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    print('Training Random Forest...')
    rf.fit(X_train, train_labels)
    rf_pred = rf.predict(X_test)
    evaluate_and_print(test_labels, rf_pred, 'Random Forest')

    # SVM (with probability via calibration)
    print('Training SVM...')
    svm_base = LinearSVC()
    svm = CalibratedClassifierCV(svm_base)
    svm.fit(X_train, train_labels)
    svm_pred = svm.predict(X_test)
    evaluate_and_print(test_labels, svm_pred, 'SVM')

    models_dir = Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec_bundle.vectorizer, models_dir / 'tfidf_vectorizer.joblib')
    joblib.dump(nb, models_dir / 'nb_model.joblib')
    joblib.dump(rf, models_dir / 'rf_model.joblib')
    joblib.dump(svm, models_dir / 'svm_model.joblib')

    # Choose best by simple accuracy among these three
    scores = {
        'nb': accuracy_score(test_labels, nb_pred),
        'rf': accuracy_score(test_labels, rf_pred),
        'svm': accuracy_score(test_labels, svm_pred),
    }
    best_name = max(scores, key=scores.get)
    best_model = {'nb': nb, 'rf': rf, 'svm': svm}[best_name]
    joblib.dump(best_model, models_dir / 'best_model.joblib')
    print('Best model:', best_name)


if __name__ == '__main__':
    # Allow running as a script: python -m detector.ml.train_ml
    from django.conf import settings as _settings  # noqa
    run()


