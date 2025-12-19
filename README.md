Kredi Kartı Müşteri Kaybı (Churn) Tahmini
Makine Öğrenmesi Uygulaması
1. Problem Tanımı

Bu çalışmanın amacı, bankacılık sektöründe müşteri kaybını (churn) önceden tahmin edebilecek bir makine öğrenmesi modeli geliştirmektir.

Müşteri kaybı, şirketlerin kârlılığını doğrudan etkileyen önemli bir problemdir. Mevcut müşteriyi elde tutmanın maliyeti, yeni müşteri kazanmaya kıyasla daha düşüktür. Bu nedenle churn tahmini, stratejik karar alma süreçlerinde kritik rol oynamaktadır.

2. Veri Seti Hakkında Bilgi

Veri seti 10.000 gözlem ve 11 değişkenden oluşmaktadır.

Değişkenler:

CreditScore (Kredi Skoru)

Geography (Ülke)

Gender (Cinsiyet)

Age (Yaş)

Tenure (Şirkette Kalma Süresi – Ay)

Balance (Bakiye)

NumOfProducts (Ürün Sayısı)

HasCrCard (Kredi Kartı Var mı)

IsActiveMember (Aktif Üye mi)

EstimatedSalary (Tahmini Maaş)

Exited (Churn – Hedef Değişken)

Veri setinde eksik gözlem bulunmamaktadır.

3. Keşifsel Veri Analizi (EDA)
3.1 Kategorik Değişkenler

Ülke Dağılımı

Fransa: %50

Almanya: %25

İspanya: %25

En yüksek churn oranı Almanya’dadır.

Cinsiyet

Erkek müşteriler çoğunluktadır.

Kadın müşterilerin churn oranı daha yüksektir.

Ürün Sayısı

2 ürün kullanan müşteriler en düşük churn oranına sahiptir.

3 ve 4 ürün kullanan müşterilerde churn riski daha yüksektir.

Aktif Üyelik

Aktif olmayan müşterilerde churn olasılığı belirgin şekilde yüksektir.

Hedef Değişken

%79,6: Churn Yok

%20,4: Churn Var

Veri seti dengesizdir.

3.2 Sayısal Değişkenler

Kredi Skoru

Yaklaşık normal dağılım göstermektedir.

Yaş

Ortalama yaş ≈ 39

Yaş arttıkça churn olasılığı artmaktadır.

Bakiye

Çok sayıda müşteri sıfır bakiyeye sahiptir.

Yüksek bakiyeli müşterilerde churn daha fazladır.

Tahmini Maaş

Uniforma yakın dağılım

Churn ile ilişkisi zayıftır.

4. Korelasyon Analizi

Değişkenler arasındaki korelasyonlar genel olarak düşüktür.

Churn ile en ilişkili değişkenler:

Yaş (pozitif ilişki)

Bakiye (zayıf pozitif ilişki)

5. Özellik Mühendisliği (Feature Engineering)

Model performansını artırmak için yeni değişkenler üretilmiştir:

Bakiye Durumu (Yok, Düşük, Orta, Yüksek, Çok Yüksek)

Kredi Skoru Durumu (Düşük, Orta, Yüksek)

Müşteri Segmenti (Premium, Standart, Riskli)

Müşteri Kıdem Kategorisi

Ortalama Aylık Gelir

Kategorik değişkenler için:

Label Encoding

One-Hot Encoding

Sayısal değişkenler için:

Z-Score Standardizasyonu

6. Modelleme Süreci
Kullanılan Modeller

Lojistik Regresyon

KNN

Karar Ağaçları

Random Forest

LightGBM

XGBoost

CatBoost

AdaBoost

Naive Bayes

Gradient Boosting

Sınıf dengesizliği nedeniyle SMOTE uygulanmıştır.

7. En İyi Model: LightGBM (SMOTE Sonrası)
Performans Sonuçları:

Accuracy (Test): %90

Recall (Churn): %89

F1-Score: %90

AUC: %96

Model, hem churn hem churn olmayan sınıflar için dengeli ve güçlü performans göstermiştir.

8. Model Yorumlanabilirliği (SHAP)

En önemli değişkenler:

Yaş

Ürün Sayısı

Aktif Üyelik

Bakiye

Ülke

Şirkette Kalma Süresi

46–65 yaş arası ve aktif olmayan müşteriler, churn açısından en riskli gruptur.

9. İş İçgörüleri

Yaşlı müşterilere özel sadakat kampanyaları

Aktif olmayan müşterileri yeniden kazanma stratejileri

Az ürün kullanan müşterilere çapraz satış

Almanya özelinde müşteri tutundurma politikaları

10. Kullanılan Teknolojiler

Python

Pandas, NumPy

Scikit-learn

LightGBM, XGBoost, CatBoost

SHAP

Matplotlib, Seaborn

11. Kaynakça

İST405 Veri Madenciliği Ders Notları

GeeksforGeeks

Medium (SMOTE ve ML makaleleri)

Plotly & Data Visualization
