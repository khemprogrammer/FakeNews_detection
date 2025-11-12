from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer


def _ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


_ensure_nltk()


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def tokenize_lemmatize(text: str) -> List[str]:
    tokens = word_tokenize(text)
    result: List[str] = []
    for tok in tokens:
        t = tok.lower()
        if t.isalpha() and t not in stop_words:
            result.append(lemmatizer.lemmatize(t))
    return result


@dataclass
class VectorizerBundle:
    vectorizer: TfidfVectorizer


def build_tfidf(texts: List[str]) -> Tuple[VectorizerBundle, any]:
    # Use min_df=1 for small datasets, can be adjusted for larger datasets
    min_df_val = 1 if len(texts) < 20 else 2
    vectorizer = TfidfVectorizer(tokenizer=tokenize_lemmatize, ngram_range=(1, 2), min_df=min_df_val)
    X = vectorizer.fit_transform(texts)
    return VectorizerBundle(vectorizer=vectorizer), X


def transform_tfidf(bundle: VectorizerBundle, texts: List[str]):
    return bundle.vectorizer.transform(texts)


