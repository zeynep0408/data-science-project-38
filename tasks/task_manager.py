"""
DS-38 — Film Yorumu Duygu Analizi (TF-IDF + Logistic Regression)
Bir streaming / e-ticaret şirketinde data scientist'sin. Kullanıcı yorumlarını
otomatik olarak pozitif (pos) / negatif (neg) etiketleyen bir duygu analizi
(sentiment analysis) modeli kuruyorsun. TF-IDF + Logistic Regression — NLP'nin
klasik ve hâlâ çok güçlü baseline'ı.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""
from os import pipe

from os import pipe

import nltk
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

# 1. NLTK movie_reviews corpus'unu indir ve (text, label) listesi döndür
def fetch_reviews():
    nltk.download('movie_reviews', quiet=True)
    from nltk.corpus import movie_reviews
    data = []
    for fid in movie_reviews.fileids():
        label = movie_reviews.categories(fid)[0]   # 'pos' veya 'neg'
        text = movie_reviews.raw(fid)
        data.append((text, label))
    return data
    pass


# 2. (text, label) listesini DataFrame olarak yükle
def load_reviews(data=None):
    if data is None:
        data = fetch_reviews()
    return pd.DataFrame(data, columns=['text', 'label'])
    pass


# 3. Veriyi keşfet — label dağılımı
def explore_data(df):
    counts = df['label'].value_counts().to_dict()
    total = len(df)
    pos = int(counts.get('pos', 0))
    neg = int(counts.get('neg', 0))
    return {
        'total': total,
        'pos_count': pos,
        'neg_count': neg,
        'balance': pos / total,
    }

    pass


# 4. Label'ları encode et (pos/neg → 1/0)
def encode_labels(df):
    out = df.copy()
    out['target'] = out['label'].map({'pos': 1, 'neg': 0}).astype(int)
    return out
    pass


# 5. Train/test split (stratified)
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pass


# 6. TfidfVectorizer + LogisticRegression pipeline kur
def build_pipeline():
       return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000)),
    ])

pass


# 7. Modeli eğit
def train_model(pipe, X_train, y_train):
    pipe.fit(X_train, y_train)
    return pipe
    pass


# 8. Modeli değerlendir
def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }
    pass


# 9. Tek bir yorum için tahmin yap
def predict_review(pipe, text):
    proba = float(pipe.predict_proba([text])[0, 1])
    pred = int(pipe.predict([text])[0])
    return {
        'prediction': pred,
        'label': 'pos' if pred == 1 else 'neg',
        'pos_probability': proba,
    }

    pass


# 10. En "pozitif" kelimeler
def top_positive_words(pipe, n=10):
    tfidf = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_idx = np.argsort(coefs)[::-1][:n]
    return [feature_names[i] for i in top_idx]
    pass


# 11. En "negatif" kelimeler
def top_negative_words(pipe, n=10):
    tfidf = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_idx = np.argsort(coefs)[:n]
    return [feature_names[i] for i in top_idx]
    pass


# 12. BoW vs TF-IDF karşılaştırması
def compare_bow_vs_tfidf(X_train, X_test, y_train, y_test):
    results = {}
    vectorizers = [
        ('bow_f1', CountVectorizer(stop_words='english')),
        ('tfidf_f1', TfidfVectorizer(stop_words='english')),
    ]
    for key, vec in vectorizers:
        pipe = Pipeline([
            ('vec', vec),
            ('clf', LogisticRegression(max_iter=1000)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[key] = f1_score(y_test, y_pred, zero_division=0)
    return results

    
    pass


# 13. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    # 1-2. Veri çek + yükle
    df = load_reviews()

    # 3. Keşif
    info = explore_data(df)

    # 4. Encode
    df = encode_labels(df)

    # 5. Split (X = yorum metni, y = target)
    X_train, X_test, y_train, y_test = split_data(df['text'], df['target'])

    # 6-7. Pipeline + eğit
    pipe = build_pipeline()
    pipe = train_model(pipe, X_train, y_train)

    # 8. Değerlendir
    metrics = evaluate_model(pipe, X_test, y_test)

    # 9-10. Örnek tahminler + en pozitif kelime
    pos_words = top_positive_words(pipe, n=10)
    sample_pos = predict_review(
        pipe,
        "An absolutely wonderful film, brilliant acting and a perfect script. I loved every minute.",
    )
    sample_neg = predict_review(
        pipe,
        "A boring, terrible waste of time. The worst movie I have seen, awful and dull.",
    )

    return {
        'balance': info['balance'],
        'test_f1': metrics['f1'],
        'top_positive_word': pos_words[0],
        'sample_pos_pred': sample_pos,
        'sample_neg_pred': sample_neg,
    }

    pass


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Pos/Neg dengesi    : {result['balance']:.2%}")
    print(f"  Test F1            : {result['test_f1']:.4f}")
    print(f"  En pozitif kelime  : {result['top_positive_word']}")
    print(f"  Pozitif örnek      : {result['sample_pos_pred']['label']} "
          f"(p={result['sample_pos_pred']['pos_probability']:.3f})")
    print(f"  Negatif örnek      : {result['sample_neg_pred']['label']} "
          f"(p={result['sample_neg_pred']['pos_probability']:.3f})")
