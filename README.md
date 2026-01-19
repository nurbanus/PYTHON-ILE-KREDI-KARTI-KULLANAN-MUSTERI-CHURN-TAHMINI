## 📄 Proje Raporu
[PDF Raporunu Görüntüle](docs/kredi_karti_churn_tahmin_rapor.pdf)


# Kredi Kartı Musteri Kaybi (Churn) Tahmini

## Makine Ogrenmesi Uygulamasi

### 1. Problem Tanimi

Bu calismanin amaci, bankacilik sektorunde musteri kaybini (churn) onceden tahmin edebilecek bir makine ogrenmesi modeli gelistirmektir.

Musteri kaybi, sirketlerin karliligini dogrudan etkileyen onemli bir problemdir. Mevcut musteriyi elde tutmanin maliyeti, yeni musteri kazanmaya kiyasla daha dusuktur. Bu nedenle churn tahmini, stratejik karar alma sureclerinde kritik rol oynamaktadir.

### 2. Veri Seti Hakkinda Bilgi

Veri seti 10000 gozlem ve 11 degiskenden olusmaktadir.

Degiskenler

CreditScore (Kredi Skoru)

Geography (Ulke)

Gender (Cinsiyet)

Age (Yas)

Tenure (Sirkette Kalma Suresi - Ay)

Balance (Bakiye)

NumOfProducts (Urun Sayisi)

HasCrCard (Kredi Karti Var mi)

IsActiveMember (Aktif Uye mi)

EstimatedSalary (Tahmini Maas)

Exited (Churn - Hedef Degisken)

Veri setinde eksik gozlem bulunmamaktadir.

### 3. Kesifsel Veri Analizi (EDA)
#### 3.1 Kategorik Degiskenler
##### Ulke Dagilimi

Fransa: %50

Almanya: %25

Ispanya: %25

En yuksek churn orani Almanya’dadir

##### Cinsiyet

Erkek musteriler sayica daha fazladir

Kadin musterilerin churn orani daha yuksektir

##### Urun Sayisi

2 urun kullanan musteriler en dusuk churn oranina sahiptir

3 ve 4 urun kullanan musterilerde churn riski daha yuksektir

##### Aktif Uyelik

Aktif olmayan musterilerde churn olasiligi daha yuksektir

##### Hedef Degisken (Churn)

Churn Yok: %79.6

Churn Var: %20.4

Veri seti dengesizdir

#### 3.2 Sayisal Degiskenler

Kredi skoru yaklasik normal dagilim gostermektedir

Yas ortalamasi yaklasik 39’dur

Yas arttikca churn olasiligi artmaktadir

Yuksek bakiyeli musterilerde churn daha fazladir

Tahmini maas ile churn arasinda guclu bir iliski bulunmamaktadir

### 4. Korelasyon Analizi

Degiskenler arasindaki korelasyonlar genellikle dusuktur

Churn ile en iliskili degiskenler:

Yas

Bakiye

### 5. Ozellik Muhendisligi (Feature Engineering)

Model performansini artirmak amaciyla yeni degiskenler uretilmistir.

Bakiye Durumu

Kredi Skoru Durumu

Musteri Segmenti

Musteri Kidem Kategorisi

Ortalama Aylik Gelir

Kategorik degiskenler icin:

Label Encoding

One Hot Encoding

Sayisal degiskenler icin:

Z Score standardizasyonu

### 6. Modelleme Sureci
Kullanilan Modeller

Lojistik Regresyon

KNN

Karar Agaclari

Random Forest

LightGBM

XGBoost

CatBoost

AdaBoost

Naive Bayes

Gradient Boosting

Sinif dengesizligi nedeniyle SMOTE uygulanmistir.

### 7. En Iyi Model

LightGBM modeli, SMOTE uygulamasi sonrasinda en iyi performansi gostermistir.

Performans Sonuclari

Accuracy (Test): %90

Recall (Churn): %89

F1 Score: %90

AUC: %96

Model, churn ve churn olmayan siniflar icin dengeli bir performans sergilemistir.

### 8. Model Yorumlanabilirligi (SHAP)

En onemli degiskenler:

Yas

Urun Sayisi

Aktif Uyelik

Bakiye

Ulke

Sirkette Kalma Suresi

46-65 yas arasi ve aktif olmayan musteriler churn acisindan en riskli gruptur.

### 9. Is Icgoruleri

Yasli musterilere ozel sadakat kampanyalari uygulanabilir

Aktif olmayan musteriler icin geri kazanma stratejileri gelistirilebilir

Az urun kullanan musterilere capraz satis yapilabilir

Almanya icin ozel musteri tutundurma politikalari olusturulabilir

### 10. Kullanilan Teknolojiler

* Python

* Pandas

* NumPy

* Scikit Learn

* LightGBM

* XGBoost

* CatBoost

* SHAP

* Matplotlib

* Seaborn
### 11.Kaynakça
* İST405 Veri Madenciliği Ders Notları

* GeeksforGeeks

* Medium (SMOTE ve Makine Öğrenmesi Makaleleri)

* Plotly & Data Visualization

