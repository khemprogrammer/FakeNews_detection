from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
from django.conf import settings


@dataclass
class Predictor:
    tfidf_vectorizer: Any | None
    ml_models: Dict[str, Any]
    lstm_model: Any | None
    lstm_tokenizer_data: Dict[str, Any] | None
    bert_tokenizer: Any | None
    bert_model: Any | None


def _load_joblib(path: Path):
    return joblib.load(path) if path.exists() else None


def load_predictor() -> Predictor:
    models_dir = Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    tfidf = _load_joblib(models_dir / 'tfidf_vectorizer.joblib')
    ml_models = {}
    for name in ['nb', 'rf', 'svm', 'best']:
        obj = _load_joblib(models_dir / f'{name}_model.joblib')
        if obj is not None:
            ml_models[name] = obj

    # Lazy imports for heavy deps
    lstm_model = None
    lstm_tokenizer_data = None
    try:
        from tensorflow import keras  # type: ignore
        lstm_path = models_dir / 'lstm_model.keras'
        tokenizer_path = models_dir / 'lstm_tokenizer.joblib'
        if lstm_path.exists():
            lstm_model = keras.models.load_model(lstm_path)
            if tokenizer_path.exists():
                lstm_tokenizer_data = joblib.load(tokenizer_path)
    except Exception:
        lstm_model = None
        lstm_tokenizer_data = None

    bert_tokenizer = None
    bert_model = None
    try:
        # Workaround for regex circular import: import regex first using importlib
        import sys
        import importlib.util
        
        # Try to import regex in a way that avoids circular imports
        if 'regex' not in sys.modules:
            try:
                spec = importlib.util.find_spec('regex')
                if spec and spec.loader:
                    importlib.util.module_from_spec(spec)
            except Exception:
                pass
        
        # Now try importing transformers
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification  # type: ignore
        bert_dir = models_dir / 'bert'
        if (bert_dir / 'config.json').exists():
            bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir.as_posix())
            bert_model = TFAutoModelForSequenceClassification.from_pretrained(bert_dir.as_posix())
    except ImportError as e:
        error_msg = str(e).lower()
        if 'regex' in error_msg or '_regex' in error_msg:
            # Regex circular import issue - try one more time with a clean import
            import traceback
            try:
                # Clear any partial regex imports
                modules_to_remove = [k for k in sys.modules.keys() if 'regex' in k.lower()]
                for mod in modules_to_remove:
                    del sys.modules[mod]
                
                # Force fresh import
                import regex  # noqa: F401
                from transformers import AutoTokenizer, TFAutoModelForSequenceClassification  # type: ignore
                bert_dir = models_dir / 'bert'
                if (bert_dir / 'config.json').exists():
                    bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir.as_posix())
                    bert_model = TFAutoModelForSequenceClassification.from_pretrained(bert_dir.as_posix())
            except Exception as retry_error:
                print("=" * 60)
                print("ERROR: Regex circular import - BERT model will not be available")
                print("This is a known issue. To fix, restart your Django server after running:")
                print("  .venv\\Scripts\\python.exe -m pip uninstall regex -y")
                print("  .venv\\Scripts\\python.exe -m pip install regex")
                print("=" * 60)
                bert_tokenizer = None
                bert_model = None
        else:
            bert_tokenizer = None
            bert_model = None
    except Exception as e:
        import traceback
        print(f"Warning: Could not load BERT model: {e}")
        traceback.print_exc()
        bert_tokenizer = None
        bert_model = None

    return Predictor(
        tfidf_vectorizer=tfidf,
        ml_models=ml_models,
        lstm_model=lstm_model,
        lstm_tokenizer_data=lstm_tokenizer_data,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
    )


def predict_text(predictor: Predictor, text: str, model_name: str = 'best') -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {'label': 'error', 'score': 0.0, 'model': 'none', 'error': 'Empty text provided'}
    
    # Prefer BERT if requested and available
    if model_name.lower() == 'bert' and predictor.bert_model and predictor.bert_tokenizer:
        try:
        import tensorflow as tf  # type: ignore
        inputs = predictor.bert_tokenizer(text, return_tensors='tf', truncation=True, padding=True)
        outputs = predictor.bert_model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
        label_idx = int(probs.argmax())
        label = 'fake' if label_idx == 1 else 'real'
        return {'label': label, 'score': float(probs[label_idx]), 'model': 'bert'}
        except Exception as e:
            return {'label': 'error', 'score': 0.0, 'model': 'bert', 'error': f'BERT prediction failed: {str(e)}'}

    # LSTM path
    if model_name.lower() == 'lstm' and predictor.lstm_model and predictor.lstm_tokenizer_data:
        try:
            from tensorflow import keras  # type: ignore
            import numpy as np  # type: ignore
            tokenizer = predictor.lstm_tokenizer_data['tokenizer']
            max_len = predictor.lstm_tokenizer_data['max_len']
            seq = tokenizer.texts_to_sequences([text])
            padded = keras.utils.pad_sequences(seq, maxlen=max_len)
            pred = predictor.lstm_model.predict(padded, verbose=0)[0][0]
            score = float(pred)
            label = 'fake' if score >= 0.5 else 'real'
            return {'label': label, 'score': max(score, 1.0 - score), 'model': 'lstm'}
        except Exception as e:
            return {'label': 'error', 'score': 0.0, 'model': 'lstm', 'error': str(e)}

    # If LSTM explicitly requested but unavailable, return a targeted error
    if model_name.lower() == 'lstm' and not (predictor.lstm_model and predictor.lstm_tokenizer_data):
        return {
            'label': 'error',
            'score': 0.0,
            'model': 'lstm',
            'error': 'LSTM model not available. Train it via "python manage.py train_lstm" to create data/models/lstm_model.keras and data/models/lstm_tokenizer.joblib.'
        }

    # ML path with TF-IDF
    vec = predictor.tfidf_vectorizer
    # Try to get the requested model, fallback to 'best', then any available model
    model = None
    model_key = None
    if model_name.lower() in ['best', 'nb', 'rf', 'svm']:
        if model_name.lower() in predictor.ml_models:
            model = predictor.ml_models[model_name.lower()]
            model_key = model_name.lower()
        elif 'best' in predictor.ml_models:
            model = predictor.ml_models['best']
            model_key = 'best'
        elif predictor.ml_models:
            model_key = list(predictor.ml_models.keys())[0]
            model = predictor.ml_models[model_key]
    
    if vec is None:
        return {'label': 'error', 'score': 0.0, 'model': 'none', 'error': 'TF-IDF vectorizer not loaded. Please train models first.'}
    if model is None:
        available = list(predictor.ml_models.keys())
        if predictor.lstm_model and predictor.lstm_tokenizer_data:
            available.append('lstm')
        if predictor.bert_model and predictor.bert_tokenizer:
            available.append('bert')
        return {'label': 'error', 'score': 0.0, 'model': 'none', 'error': f'Model {model_name} not found. Available: {available}'}
    
    try:
        X = vec.transform([text])
        proba = getattr(model, 'predict_proba', None)
        if proba:
            probs = proba(X)[0]
            # Get class predictions
            if hasattr(model, 'classes_'):
                classes = model.classes_
                # Find index of fake class - handle numpy array
                fake_idx = None
                real_idx = None
                for i, cls in enumerate(classes):
                    cls_str = str(cls).lower().strip()
                    if cls_str == 'fake':
                        fake_idx = i
                    elif cls_str == 'real':
                        real_idx = i
                
                # Get probability for fake class
                if fake_idx is not None:
                    fake_prob = float(probs[fake_idx])
                    real_prob = float(probs[real_idx]) if real_idx is not None else 1.0 - fake_prob
                else:
                    # Fallback: assume first class is fake or use index
                    fake_prob = float(probs[1] if len(probs) > 1 else probs[0])
                    real_prob = 1.0 - fake_prob
                
                # Determine label based on which probability is higher
                if fake_prob >= real_prob:
                    label = 'fake'
                    score = fake_prob
                else:
                    label = 'real'
                    score = real_prob
            else:
                # Fallback if no classes_ attribute
                idx = int(probs.argmax())
                label = 'fake' if idx == 1 else 'real'
                score = float(probs[idx])
        else:
            pred = model.predict(X)[0]
            pred_str = str(pred).lower()
            if pred_str in ['fake', 'real']:
                label = pred_str
                score = 0.75
            else:
                # Try to get probability if available
                label = 'fake' if pred_str in ['1', 'true'] else 'real'
                score = 0.75 if label == 'fake' else 0.25
        
        return {'label': label, 'score': float(score), 'model': model_key or 'unknown'}
    except Exception as e:
        return {'label': 'error', 'score': 0.0, 'model': model_key or 'unknown', 'error': str(e)}


