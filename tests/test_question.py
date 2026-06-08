import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    fetch_reviews, load_reviews, explore_data, encode_labels,
    split_data, build_pipeline, train_model, evaluate_model,
    predict_review, top_positive_words, top_negative_words,
    compare_bow_vs_tfidf, run_pipeline,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar indirme/eğitme yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """İlk testte NLTK movie_reviews indirir, sonraki tüm testler cache'den okur."""
    return load_reviews()


@pytest.fixture(scope="module")
def split(raw_df):
    df = encode_labels(raw_df)
    return split_data(df['text'], df['target'])


@pytest.fixture(scope="module")
def trained_pipeline(split):
    X_train, X_test, y_train, y_test = split
    pipe = build_pipeline()
    return train_model(pipe, X_train, y_train)


# 1. fetch_reviews
def test_fetch_reviews():
    data = fetch_reviews()
    assert len(data) == 2000
    # her eleman (text, label) ikilisi
    text, label = data[0]
    assert isinstance(text, str) and len(text) > 0
    assert label in {'pos', 'neg'}


# 2. load_reviews
def test_load_reviews_shape(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert raw_df.shape == (2000, 2)
    assert list(raw_df.columns) == ['text', 'label']


# 3. load — label değerleri
def test_load_reviews_labels(raw_df):
    assert set(raw_df['label'].unique()) == {'pos', 'neg'}


# 4. explore_data
def test_explore_data_structure(raw_df):
    info = explore_data(raw_df)
    assert set(info.keys()) >= {'total', 'pos_count', 'neg_count', 'balance'}
    assert info['total'] == 2000
    assert info['pos_count'] == 1000
    assert info['neg_count'] == 1000
    assert info['pos_count'] + info['neg_count'] == info['total']
    # Tam dengeli veri → balance ~0.5
    assert abs(info['balance'] - 0.5) < 0.01


# 5. encode_labels
def test_encode_labels_binary(raw_df):
    df = encode_labels(raw_df)
    assert 'target' in df.columns
    assert set(df['target'].unique()) == {0, 1}
    assert df['target'].dtype in (np.int64, np.int32, int)
    # pos → 1, neg → 0
    assert df[df['label'] == 'pos']['target'].iloc[0] == 1
    assert df[df['label'] == 'neg']['target'].iloc[0] == 0


# 6. split_data
def test_split_data_stratified(raw_df):
    df = encode_labels(raw_df)
    X_train, X_test, y_train, y_test = split_data(df['text'], df['target'])
    # 80/20
    assert abs(len(X_train) / len(df) - 0.8) < 0.01
    # Stratify: train ve test pos oranı yakın olmalı
    assert abs(y_train.mean() - y_test.mean()) < 0.01


# 7. build_pipeline
def test_build_pipeline_type():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    names = dict(pipe.steps)
    assert 'tfidf' in names and 'clf' in names
    assert isinstance(names['tfidf'], TfidfVectorizer)
    assert isinstance(names['clf'], LogisticRegression)
    assert names['tfidf'].stop_words == 'english'


# 8. train_model
def test_train_model_fits(trained_pipeline, split):
    X_train, X_test, y_train, y_test = split
    preds = trained_pipeline.predict(X_test[:5])
    assert len(preds) == 5


# 9. evaluate_model
def test_evaluate_model(trained_pipeline, split):
    _, X_test, _, y_test = split
    res = evaluate_model(trained_pipeline, X_test, y_test)
    assert set(res.keys()) >= {
        'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'
    }
    assert 0 <= res['accuracy'] <= 1
    assert res['confusion_matrix'].shape == (2, 2)
    # TF-IDF + LogReg film yorumlarında güçlü baseline
    assert res['f1'] > 0.78


# 10. predict_review — pozitif
def test_predict_review_positive(trained_pipeline):
    out = predict_review(
        trained_pipeline,
        "A wonderful, brilliant film with excellent acting and a perfect story. I loved it.",
    )
    assert set(out.keys()) >= {'prediction', 'label', 'pos_probability'}
    assert out['prediction'] == 1
    assert out['label'] == 'pos'
    assert 0 <= out['pos_probability'] <= 1
    assert out['pos_probability'] > 0.5


# 11. predict_review — negatif
def test_predict_review_negative(trained_pipeline):
    out = predict_review(
        trained_pipeline,
        "A boring and terrible movie, the worst waste of time. Awful, dull and stupid.",
    )
    assert out['prediction'] == 0
    assert out['label'] == 'neg'
    assert out['pos_probability'] < 0.5


# 12. top_positive_words
def test_top_positive_words(trained_pipeline):
    words = top_positive_words(trained_pipeline, n=10)
    assert isinstance(words, list)
    assert len(words) == 10
    assert all(isinstance(w, str) for w in words)
    # Pozitif-tipik kelimelerden en az biri olmalı
    positive = {'great', 'excellent', 'perfect', 'wonderful', 'best',
                'brilliant', 'memorable', 'amazing', 'enjoyable', 'fun',
                'terrific', 'outstanding', 'powerful'}
    assert len(set(w.lower() for w in words) & positive) >= 1


# 13. top_negative_words
def test_top_negative_words(trained_pipeline):
    words = top_negative_words(trained_pipeline, n=10)
    assert isinstance(words, list)
    assert len(words) == 10
    # Negatif-tipik kelimelerden en az biri olmalı
    negative = {'bad', 'worst', 'boring', 'waste', 'stupid', 'awful',
                'terrible', 'poor', 'dull', 'unfortunately', 'ridiculous',
                'lame', 'mess', 'nothing'}
    assert len(set(w.lower() for w in words) & negative) >= 1
    # Pozitif ve negatif kelimeler farklı olmalı
    pos_words = top_positive_words(trained_pipeline, n=10)
    assert set(words) != set(pos_words)


# 14. compare_bow_vs_tfidf
def test_compare_bow_vs_tfidf(split):
    X_train, X_test, y_train, y_test = split
    cmp = compare_bow_vs_tfidf(X_train, X_test, y_train, y_test)
    assert set(cmp.keys()) >= {'bow_f1', 'tfidf_f1'}
    # İkisi de iyi performans göstermeli
    assert cmp['bow_f1'] > 0.75
    assert cmp['tfidf_f1'] > 0.75


# 15. run_pipeline
def test_run_pipeline_full():
    result = run_pipeline()
    assert set(result.keys()) >= {
        'balance', 'test_f1', 'top_positive_word',
        'sample_pos_pred', 'sample_neg_pred'
    }
    assert result['test_f1'] > 0.78
    assert abs(result['balance'] - 0.5) < 0.01
    # Örnek tahminler doğru
    assert result['sample_pos_pred']['prediction'] == 1
    assert result['sample_neg_pred']['prediction'] == 0
    # En pozitif kelime mantıklı
    assert isinstance(result['top_positive_word'], str)
    assert len(result['top_positive_word']) > 0


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
