# Data Science Project 38 — Film Yorumu Duygu Analizi (TF-IDF + Logistic Regression)

**Modül**: ML-04 (Sınıflandırma 2 / NLP) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir streaming / e-ticaret şirketinde **data scientist** olarak çalışıyorsun. Platforma her gün binlerce kullanıcı yorumu düşüyor: "harika bir film, bayıldım" ya da "berbat, zaman kaybı". Ürün ekibi bu yorumları tek tek elle okuyamıyor. Senden, gelen her yorumu **pozitif mi negatif mi** olduğunu otomatik etiketleyen bir **duygu analizi (sentiment analysis)** modeli kurmanı istiyorlar.

Senin görevin: bir yorum metnine bakıp **olumlu (pos)** mu **olumsuz (neg)** mu olduğunu tahmin eden bir model kurmak. Doğru çalışırsa negatif yorumlar otomatik flag'lenip destek ekibine düşer, pozitif yorumlar öne çıkarılır.

Bunun için NLP'nin klasik ve hâlâ çok güçlü baseline'ını kullanacaksın: **TF-IDF + Logistic Regression**. TF-IDF her kelimeyi sadece saymakla kalmaz, **o kelimenin ne kadar ayırt edici olduğunu** da ölçer (her yorumda geçen "movie" gibi kelimeler bastırılır, "brilliant" / "worst" gibi nadir ama belirleyici kelimeler öne çıkar). Logistic Regression de bu vektörlerden hangi kelimelerin pozitife hangilerinin negatife işaret ettiğini öğrenir.

Veri **tam dengeli**: 1000 pozitif, 1000 negatif yorum (%50/%50). Yani burada accuracy de F1 de güvenilir; dengesizlik problemi yok, asıl odak **metni sayıya çevirme** ve **doğru kelimeleri yakalama**.

Bu projede ML-04 dersinde öğrendiklerini birleştirip uygulayacaksın:
- ✅ **NLTK corpus çekme** (`nltk.download('movie_reviews')`)
- ✅ **Metin → sayı** dönüşümü (`TfidfVectorizer` — TF-IDF)
- ✅ **LogisticRegression** (metin sınıflandırmanın güçlü baseline'ı)
- ✅ **stop_words** (anlamsız kelimeleri eleme)
- ✅ **Stratified train/test split**
- ✅ **Precision / Recall / F1**
- ✅ **`coef_` ile "en pozitif / en negatif kelimeler"**
- ✅ **BoW vs TF-IDF** karşılaştırması
- ✅ **sklearn Pipeline** (vektörize + model tek akışta)

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-38

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 718 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Dataset — NLTK Movie Reviews Corpus

### Kaynak
- **NLTK `movie_reviews` corpus** — Pang & Lee (2004), "polarity dataset v2.0"
- Akademik kaynak: Pang, B., & Lee, L. (2004). *A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts.* Proceedings of the ACL.
- IMDb'den derlenmiş 2000 film yorumu (yorumun yıldız sayısına göre pos/neg etiketlenmiş).

### Veri Çekme Yöntemi (önemli)
Veri seti **repo'da YOK** — URL'den manuel indirmiyorsun, **NLTK kendi cache'ine** indiriyor. Bu yüzden `fetch_reviews()` fonksiyonun:

1. `nltk.download('movie_reviews', quiet=True)` çağırır (ilk seferde indirir, sonra `~/nltk_data`'dan okur — cache)
2. `from nltk.corpus import movie_reviews`
3. `movie_reviews.fileids()` ile 2000 dosyayı dolaşır; her dosya için `movie_reviews.categories(fid)` ('pos'/'neg') ve `movie_reviews.raw(fid)` (ham metin) alır
4. `(text, label)` ikililerinden bir liste döndürür

> **Not:** İlk testte internet gerekir (NLTK paketi indirilir, ~3 MB). Sonraki tüm çalıştırmalar lokal cache'den okur — URL / SSL ile uğraşmana gerek yok, NLTK halleder.

### Boyut & Target
- **2000 yorum × 2 sütun** (`text`, `label`)
- Target: `label` (`pos` / `neg`) — yorum olumlu mu olumsuz mu?
- Class dağılımı:
  - `pos`: 1000 (%50)
  - `neg`: 1000 (%50)
- **Tam dengeli** — accuracy de F1 de güvenilir.

### Dosya Formatı
- NLTK corpus'u olarak gelir; sen `pd.DataFrame(data, columns=['text', 'label'])` ile DataFrame'e çevirirsin.
- `text`: ham yorum metni (uzun, birden çok cümle, küçük harf, tokenize edilmemiş ham hali kullanılabilir)
- `label`: `'pos'` veya `'neg'`

### Örnek (kısaltılmış)
```
label | text
pos   | "... a wonderful little production. the filming technique is very ... brilliant ..."
neg   | "... this movie is just plain bad. boring, terrible acting, a complete waste of time ..."
```
Pozitif yorumlarda **great**, **excellent**, **wonderful**, **brilliant**, **perfect**; negatif yorumlarda **bad**, **worst**, **boring**, **waste**, **stupid** sık görülür. Modelin tam da bu kelimeleri öğrenmesini bekliyoruz.

### Domain Notu
Yorumlar İngilizce ve uzun (paragraf seviyesinde). Tek tek kelimelerin yanı sıra TF-IDF'in **kelime ağırlıklandırması** burada işe yarar: her yorumda geçen jenerik kelimeler ("film", "movie", "story") bastırılırken, az ama belirleyici kelimeler ("masterpiece", "awful") öne çıkar. Bu yüzden **TF-IDF + LogReg** kelime sayımına (bag of words) göre genelde biraz daha iyi sonuç verir.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **13 fonksiyonu** sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`fetch_reviews()`** — `nltk.download('movie_reviews')`, `(text, label)` listesi döndür
2. **`load_reviews(data=None)`** — `pd.DataFrame ['text','label']` (2000 satır)
3. **`explore_data(df)`** — total, pos_count, neg_count, balance
4. **`encode_labels(df)`** — `label`: pos→1, neg→0 (yeni `target` sütunu)
5. **`split_data(X, y)`** — 80/20, stratify=y, random_state=42
6. **`build_pipeline()`** — TfidfVectorizer(stop_words='english') + LogisticRegression(max_iter=1000)
7. **`train_model(pipe, X_train, y_train)`** — fit + dön
8. **`evaluate_model(pipe, X_test, y_test)`** — accuracy, precision, recall, f1, confusion matrix
9. **`predict_review(pipe, text)`** — tek yorum için tahmin + pozitif olasılığı
10. **`top_positive_words(pipe, n=10)`** — `coef_` en büyük → en pozitif n kelime
11. **`top_negative_words(pipe, n=10)`** — `coef_` en küçük → en negatif n kelime
12. **`compare_bow_vs_tfidf(...)`** — CountVectorizer vs TfidfVectorizer F1 karşılaştır
13. **`run_pipeline()`** — uçtan uca akış, özet dict dön

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] **NLTK corpus** çekip DataFrame'e dönüştürebileceksin
- [x] **Metni sayıya** çevirebileceksin (`TfidfVectorizer`, TF-IDF)
- [x] **LogisticRegression** ile metin sınıflandırma yapabileceksin
- [x] **stop_words** temizliğini uygulayabileceksin
- [x] **Stratified split** ile sınıf oranını koruyabileceksin
- [x] **Precision / Recall / F1** hesaplayabileceksin
- [x] **`coef_`** ile modelin "en pozitif / en negatif kelimelerini" çıkarabileceksin
- [x] **BoW vs TF-IDF** farkını ölçebileceksin
- [x] **sklearn Pipeline** ile vektörize + model adımlarını tek akışta birleştirebileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (15 test)

Tümü pass olmalı:
- Corpus fetch (internet gerekli, ilk test indirir) + 2000 yorum
- Satır sayısı (2000) ve label değerleri doğru mu
- Pos/neg dengesi 1000/1000 (balance ~%50) tespit edilmiş mi
- `target` 0/1 encode edilmiş mi (pos=1, neg=0)
- Stratified split korunmuş mu
- Pipeline tipi doğru mu (TfidfVectorizer + LogisticRegression)
- F1 > 0.78 (model güçlü baseline)
- Pozitif/negatif örnek yorumlar doğru sınıflanıyor mu
- En pozitif kelimeler mantıklı mı ('great', 'excellent', 'perfect' gibi)
- En negatif kelimeler mantıklı mı ('bad', 'worst', 'boring' gibi)

## 📊 Beklenen Sonuçlar

```
Balance (pos oranı): ~%50.0
Test F1: ~0.80-0.85 (TF-IDF + LogReg güçlü baseline)
En pozitif kelimeler: great, excellent, perfect, wonderful, memorable ...
En negatif kelimeler: bad, worst, boring, waste, stupid, unfortunately ...
BoW vs TF-IDF: ikisi de ~0.80, TF-IDF genelde biraz önde
```

## 💡 İpuçları

- **İlk testte internet** gerekli (NLTK paketi indirme). Sonraki testler cache'den okur.
- `nltk.download('movie_reviews', quiet=True)` çağırmadan `from nltk.corpus import movie_reviews` import etme
- `movie_reviews.raw(fid)` → ham metin, `movie_reviews.categories(fid)[0]` → 'pos'/'neg'
- Pipeline ham string listesi alır; TfidfVectorizer otomatik vektörize eder — sen elle vektörize etme
- `predict_review`'da tek string değil **liste** geç: `pipe.predict_proba([text])`
- LogReg'de `clf.coef_[0]` → her kelimenin ağırlığı. **En büyük** → en pozitif, **en küçük (negatif)** → en negatif
- `np.argsort(coefs)[::-1][:n]` → en büyük n (pozitif); `np.argsort(coefs)[:n]` → en küçük n (negatif)
- `precision_score`, `recall_score`, `f1_score` — hepsi `sklearn.metrics`

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (testler fail olur)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- `data/` klasörü repo'ya **gitmez** (.gitignore'da exclude); zaten NLTK kendi cache'ine indiriyor
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
