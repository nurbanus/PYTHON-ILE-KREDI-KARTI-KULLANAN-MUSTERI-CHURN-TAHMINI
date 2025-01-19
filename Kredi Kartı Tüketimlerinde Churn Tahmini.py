#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Veri hazırlık kütüphaneleri"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""Modelleme Kütüphaneleri"""
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
"""Model Eleme"""
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

"""Diğer"""
import os
import warnings
#from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category = ConvergenceWarning)
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[2]:


df = pd.read_csv(r"C:\Users\HANDENUR\Downloads\Churn.csv")


# In[3]:


df.head(10)


# In[122]:


df.tail(10)


# In[123]:


df.shape


# 10.000 satır , 11 sütündan oluşuyor. değişken isimlerini değiştirelim.

# In[3]:


# Sütun isimlerini Türkçeye çevirme
df.rename(columns={
    "CreditScore": "kredi_skoru",
    "Geography": "ulke",
    "Gender": "cinsiyet",
    "Age": "yas",
    "Tenure": "sirkette_kaldıgı_ay_sayısı",
    "Balance": "bakiye",
    "NumOfProducts": "urun_sayısı",
    "HasCrCard": "kredi_kartı_var_mı",
    "IsActiveMember": "aktif_uye_mi",
    "EstimatedSalary": "tahmini_maas",
    "Exited": "ayrıldı_mı(churn)"
}, inplace=True)

# Yeni sütun isimlerini görüntüle
df.head()


# sadece üç ülke var. o yüzden ülke kategorik olur diye yorumladık.

# In[125]:


df.info()


# In[126]:


df.isnull().sum()


# In[127]:


df["ulke"].unique()


# eksik değer yok . veri tiplerini kontrol edelim.

# In[4]:


def grab_col_names(dataframe, cat_th=10, car_th=15):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat,cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car  


# In[5]:


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


# In[6]:


cat_cols


# In[7]:


num_cols


# In[133]:


df["urun_sayısı"].unique()


# In[136]:


cat_cols


# In[137]:


df.describe().T


# In[ ]:





# ### değişken analizleri

# Hedef Değişkenin Dağılımı Veri setimizdeki "churn" değişkeninin dağılımını analiz etmek, hem dengesiz veri problemi olup olmadığını tespit etmemizi sağlar, hem de modelimizin bu dağılımı ne kadar iyi tahmin edebileceğine dair bize bir fikir verir. Eğer hedef değişkenimiz dengesizse (örneğin, bir sınıf diğerlerine göre çok daha fazla gözleme sahipse), modelimiz bu dengesizliği öğrenebilir ve yanlış tahminlerde bulunabilir.

# In[7]:


### kategorik değişken analizi:

def cat_summary(dataframe, col_name, plot=False):
    summary = pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
    print(summary)
    print("##########################################")
    
    if plot:
        plt.figure(figsize=(6, 6))
        # Pasta grafiği
        sizes = summary[col_name]  # Her kategorinin frekansı
        labels = summary.index  # Kategoriler
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
        plt.title(f"{col_name} Dağılımı")
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)


# In[139]:


#sütun grafiği şeklinde de yapalım:
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe , palette="pink")
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True) 
    



# In[9]:


### numerik değişken analizi:
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=10)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()       

for col in num_cols:
    num_summary(df, col, plot=True)


# In[23]:


#grafiklere tek tek değil de birlikte bakmak istersek
df[num_cols].hist(figsize=(8, 10), bins=10, xlabelsize=8, ylabelsize=8, color="palevioletred")


# In[10]:


# Kredi skorunu segmentlere ayır
kopya=df.copy()
bins = [0, 500, 700, 850]  # Aralıklar: 0-500, 501-700, 701-850
labels = ['Düşük', 'Orta', 'Yüksek']
kopya['kredi_segmenti'] = pd.cut(kopya['kredi_skoru'], bins=bins, labels=labels)

# Segmentlere göre analiz
segment_analysis = kopya.groupby('kredi_segmenti').mean()

print("\nSegmentlere Göre Analiz:")
segment_analysis


# In[11]:


# Segmentlerin oranlarını hesaplama
segment_counts = kopya['kredi_segmenti'].value_counts()

# Pasta grafiği çizimi
segment_counts.plot.pie(
    autopct="%.1f%%",  # Yüzde değerlerini göster
    startangle=90,  # Grafiği 90 dereceden başlat
    colors=sns.color_palette("pastel"),  # Pastel renkler
    ylabel="",  # Y ekseni etiketi olmadan
    title="Kredi Skoru Segmentlerinin Dağılımı"
)
plt.show()


# ### KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ

# In[12]:


#kategorik değişkenlerin hedef değişkene göre analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Bağımlı değişken kategorik olduğunda, bağımsız değişkenin her bir kategorisine göre bağımlı değişkenin
    dağılımını gösteren bir özet çıkarır.

    :param dataframe: pandas DataFrame
    :param target: kategorik bağımlı değişken
    :param categorical_col: analiz edilecek kategorik bağımsız değişken
    """
 # Çapraz tablo ile bağımsız ve bağımlı değişken ilişkisini inceleme
    cross_tab = pd.crosstab(dataframe[categorical_col], dataframe[target], normalize="index") * 100
    print(f"--- {categorical_col} için Churn Dağılımı ---")
    print(cross_tab)
    print("\n")
for col in cat_cols:
    target_summary_with_cat(df, "ayrıldı_mı(churn)", col)


# In[144]:


import pandas as pd
import matplotlib.pyplot as plt

def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    """
    Bağımlı değişken kategorik olduğunda, bağımsız değişkenin her bir kategorisine göre bağımlı değişkenin
    dağılımını gösteren bir özet çıkarır ve isteğe bağlı olarak grafik çizer.

    :param dataframe: pandas DataFrame
    :param target: kategorik bağımlı değişken
    :param categorical_col: analiz edilecek kategorik bağımsız değişken
    :param plot: bool, True ise grafik çizer
    """ 
    # Çapraz tablo ile bağımsız ve bağımlı değişken ilişkisini inceleme
    cross_tab = pd.crosstab(dataframe[categorical_col], dataframe[target], normalize="index") * 100
    print(f"--- {categorical_col} için Churn Dağılımı ---")
    print(cross_tab)
    print("\n")
    
    if plot:
        cross_tab.plot(kind="bar", stacked=True, figsize=(8, 6), color=["#b0c4de", "#66B2FF"])
        plt.title(f"{categorical_col} Kategorilerine Göre {target} Dağılımı")
        plt.ylabel("Yüzde (%)")
        plt.xlabel(categorical_col)
        plt.legend(title=target)
        plt.xticks(rotation=45)
        plt.show()

# Tüm kategorik değişkenler için analiz ve grafik
for col in cat_cols:
    target_summary_with_cat(df, "ayrıldı_mı(churn)", col, plot=True)


# ### numerik değişkenlerin targeta göre analizi

# In[145]:


##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "ayrıldı_mı(churn)", col)


# In[146]:


import matplotlib.pyplot as plt

def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    """
    Sayısal bir değişkenin hedef değişkene göre özetini çıkarır ve opsiyonel olarak grafik çizer.
    
    :param dataframe: pandas DataFrame
    :param target: kategorik hedef değişken
    :param numerical_col: analiz edilecek sayısal bağımsız değişken
    :param plot: bool, True ise grafik çizer
    """
    # Hedef değişkene göre gruplama ve ortalama hesaplama
    summary = dataframe.groupby(target).agg({numerical_col: "mean"})
    print(summary, end="\n\n\n")
    
    # Grafik çizimi
    if plot:
        summary.plot(kind="bar", legend=False, color="mediumseagreen", figsize=(8, 6))
        plt.title(f"{numerical_col} Ortalamaları (Hedef Değişken: {target})")
        plt.ylabel(f"Ortalama {numerical_col}")
        plt.xlabel(target)
        plt.xticks(rotation=0)  # X ekseni etiketlerini düz yaz
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

# Tüm sayısal sütunlar için analiz ve grafik
for col in num_cols:
    target_summary_with_num(df, "ayrıldı_mı(churn)", col, plot=True)

şirketten ayrılanların kredi skoru düşmüş.
şirketten ayrılanların yaşı daha büyük.
şirketten ayrılanların mantıken şirkette kaldığı ay sayısı daha az.
şirketten ayrılanların bakiyeleri daha fazla.
şirketten ayrılanların tahmini maaşları daha fazlaymış
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax,cmap="YlGnBu")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
# In[147]:


sns.set()
corr = df[num_cols].corr()
ax = sns.heatmap(corr
                 ,center=0
                 ,annot=True
                 ,linewidths=.2
                 ,cmap="YlGnBu")
plt.show()

Korelasyon, iki değişken arasındaki doğrusal ilişkiyi ölçer. Değerler genelde -1 ile +1 arasında değişir:
+1: Pozitif güçlü ilişki
-1: Negatif güçlü ilişki
0: Hiçbir ilişki yok
Düşük Korelasyon Değerleri

Korelasyon matrisinde, tüm değişkenler arasındaki korelasyon değerleri oldukça düşük (~0.00-0.03). Bu, değişkenler arasında anlamlı bir doğrusal ilişkinin olmadığını gösteriyor.
# In[148]:


df.corr(method = 'pearson',numeric_only = True).unstack().idxmin()


# In[149]:


df.corr(method = 'pearson',numeric_only = True).unstack().idxmax()  


# beklenildiği gibi değişkenin kendisiyle olan korelasyonu 1 dir ve maxtır

# In[150]:


df[num_cols].corrwith(df["ayrıldı_mı(churn)"]).sort_values(ascending=False)


# yas en yüksek korelasyona sahip değişken olsa da, korelasyon değeri (0.285) düşük-orta düzeyde ve diğer değişkenler ise oldukça zayıf ilişkiler sergiliyor.
# Bu sonuçlar, ayrıldı_mı(churn) değişkenini tahmin etmek için tek başına doğrusal ilişkilerin yeterli olmayabileceğini gösteriyor.
# İlişkilerin düşük olması, hedef değişkeni anlamak için daha karmaşık (ör. doğrusal olmayan modeller, etkileşim terimleri) yöntemlere ihtiyaç duyulabileceğini işaret ediyor.

# ### detaylı görselleştirme

# In[153]:


from itertools import combinations

def numcols_target_corr(df, target="ayrıldı_mı(churn)"):

    
    # Sayısal sütunların ikili kombinasyonlarını oluştur
    numvar_combinations = list(combinations(num_cols, 2))
    
    # Her kombinasyon için scatter plot oluştur
    for item in numvar_combinations:
        plt.subplots(figsize=(14, 8))
        sns.scatterplot(x=df[item[0]], y=df[item[1]], hue=df[target], palette="Set2")\
            .set_title(f'{item[0]}   &   {item[1]}')
        plt.grid(True)
        plt.show()

# Fonksiyonu çağır
numcols_target_corr(df, target="ayrıldı_mı(churn)")


# In[154]:


# Sayısal - Sayısal İlişki: Scatterplot
sns.pairplot(df[num_cols], diag_kind='kde', corner=True)
plt.show()


# In[155]:


# Violin plot çizdirme

for degisken in num_cols:
        plt.figure(figsize=(8, 5))
        sns.violinplot(x="ayrıldı_mı(churn)", y=degisken, data=df, hue="ayrıldı_mı(churn)", palette=["lightpink", "thistle"], legend=False)
        plt.title(f"{degisken} ve churn Arasındaki Dağılım (Violin Plot)")
        plt.show()


# ### base model

# In[15]:


dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["ayrıldı_mı(churn)"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["ayrıldı_mı(churn)"]
X = dff.drop(["ayrıldı_mı(churn)"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


# ### aykırı değerleri inceleme

# In[12]:


def draw_boxplots_for_numeric_columns(df, figsize=(10, 5), palette="Set2"):
    """
    Tüm numerik değişkenler için kutu grafiği çizer.

    Args:
        df (pd.DataFrame): Veri çerçevesi (DataFrame).
        figsize (tuple): Grafik boyutu (default: (10, 5)).
        palette (str): Grafik renk paleti (default: "Set2").
    """

    for col in num_cols:
        plt.figure(figsize=figsize)
        sns.boxplot(data=df, x=col, palette=palette)
        plt.title(f'Boxplot for {col}')
        plt.xlabel('')
        plt.ylabel('Values')
        plt.show()
        
draw_boxplots_for_numeric_columns(df) 


# In[10]:


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
for col in num_cols:
    print(col, check_outlier(dff, col))
    


# In[13]:


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(dff, col))
    if check_outlier(dff, col):
        replace_with_thresholds(dff, col)  #2kere çalışıcak 


# In[170]:


# Aykırı Değer Analizi -test
for col in num_cols:
    print(col, check_outlier(dff,col))
    if check_outlier(dff, col):
        replace_with_thresholds(dff, col)


# In[14]:


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


# In[15]:


import matplotlib.pyplot as plt

# kredi_skoru değişkeni için boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(df["kredi_skoru"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("kredi_skoru Değişkeni Box Plot")
plt.xlabel("Kredi Skoru")
plt.show(),


# In[16]:


import pandas as pd

def calculate_outlier_ratio(dataframe, columns=None):
    """
    Verilen bir veri çerçevesindeki sayısal sütunlar için aykırı değer oranlarını hesaplar.
    
    Parameters:
        dataframe (pd.DataFrame): Veri çerçevesi.
        columns (list, optional): Kontrol edilecek sütunların listesi. None ise tüm sayısal sütunlar kullanılır.
        
    Returns:
        pd.DataFrame: Her bir sütun için aykırı değer oranını içeren bir veri çerçevesi.
    """
    
    outlier_ratios = {}
    for col in num_cols:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)).sum()
        outlier_ratios[col] = outliers / len(dataframe) * 100  # Aykırı değer oranı (%)
    
    return pd.DataFrame(outlier_ratios.items(), columns=["Column", "Outlier Ratio (%)"])

outlier_ratios = calculate_outlier_ratio(df)
print(outlier_ratios)


# ### yeni değişken üretmek

# In[9]:


df_copy = df.copy()


# In[10]:


# Yaşa göre yaşam dönemi kategorisi
#df_copy['yas_donemi'] = pd.cut(df_copy['yas'], bins=[0, 25, 40, 60, 100], labels=['Genç', 'Orta Yaş Altı', 'Orta Yaş', 'Yaşlı'])
# Bakiye durumunu sınıflandırma
df_copy['bakiye_durumu'] = pd.cut(df_copy['bakiye'], bins=[-1, 0, 50000, 100000, 200000, float('inf')], 
                                  labels=['Hiç Para Yok', 'Düşük', 'Orta', 'Yüksek', 'Çok Yüksek'])

# Aylık ortalama gelir
df_copy['ortalama_gelir'] = df_copy['tahmini_maas'] / (df_copy['sirkette_kaldıgı_ay_sayısı'] + 1)  # 0'a bölmeyi önlemek için +1

# Kredi skoru durumunu sınıflandırma
df_copy['kredi_skoru_durumu'] = pd.cut(df_copy['kredi_skoru'], bins=[0, 500, 700, 850], 
                                       labels=['Düşük', 'Orta', 'Yüksek'])
def musterı_kategorisi(row):
    if row['bakiye_durumu'] == 'Çok Yüksek' and row['kredi_skoru_durumu'] == 'Yüksek':
        return 'Premium'
    elif row['bakiye_durumu'] in ['Yüksek', 'Orta'] and row['kredi_skoru_durumu'] == 'Orta':
        return 'Standart'
    else:
        return 'Riskli'

df_copy['musteri_kategorisi'] = df_copy.apply(musterı_kategorisi, axis=1)

df_copy['musteri_kidem_kategorisi'] = pd.cut(
    df_copy['sirkette_kaldıgı_ay_sayısı'],
    bins=[0, 3, 6, 10],  # Kısa, Orta ve Uzun vadeli için eşik değerler
    labels=['Kısa Vadeli', 'Orta Vadeli', 'Uzun Vadeli'],
    right=True,  # Sağ kapsayıcılığı aç
    include_lowest=True  # En düşük değeri de kapsa
)


# In[8]:


df_copy.head()


# In[9]:


df_copy.info()


# ## one -hot & label hot encoding 

# In[10]:


# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df_copy)


# In[11]:


cat_cols


# In[11]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd
# Label Encoding yapılacak değişkenler
label_cols = ['bakiye_durumu', 'kredi_skoru_durumu', 'musteri_kategorisi', 'musteri_kidem_kategorisi']
le = LabelEncoder()

for col in label_cols:
    df_copy[col] = le.fit_transform(df_copy[col])

# One-Hot Encoding yapılacak değişkenler
df_copy = pd.get_dummies(df_copy, columns=['ulke', 'cinsiyet'], drop_first=False, dtype=int) 


# In[34]:


df_copy.head(10)


# In[44]:


y = df_copy["ayrıldı_mı(churn)"]
X = df_copy.drop(["ayrıldı_mı(churn)"], axis=1)


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


# ### veri normalizasyonu : Z-score

# In[12]:


## z-score
from scipy.stats import zscore
# Z-score hesaplanacak sayısal sütunlar 
sayisal = ['kredi_skoru', 'yas', 'sirkette_kaldıgı_ay_sayısı', 'bakiye', 
                'tahmini_maas', 'ortalama_gelir']

# Z-score hesaplama
df_copy[sayisal] = df_copy[sayisal].apply(zscore)
df_copy.head()


# In[38]:


#y = df_copy["ayrıldı_mı(churn)"]
#X = df_copy.drop(["ayrıldı_mı(churn)"], axis=1)


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


# ### MODELLER VE MODEL SEÇİMİ

# In[21]:


df_model=df_copy.copy()
#y = df_model["ayrıldı_mı(churn)"]
#X = df_model.drop([["ayrıldı_mı(churn)"]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345,stratify=y)


# ### KNN

# In[199]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# KNN modelinin tanımlanması
knn_model = KNeighborsClassifier(n_neighbors=5)  # K değeri: 5 (isteğe göre değiştirilebilir)

# Modeli eğitme
knn_model.fit(X_train, y_train)

# Tahmin yapma
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)

# Performans değerlendirme
print("Eğitim Seti Sınıflandırma Raporu:")
print(classification_report(y_train, y_train_pred))

print("Test Seti Sınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred))

# ROC-AUC Skoru
y_train_proba = knn_model.predict_proba(X_train)[:, 1]  # Eğitim seti pozitif sınıf olasılıkları
y_test_proba = knn_model.predict_proba(X_test)[:, 1]  # Test seti pozitif sınıf olasılıkları

roc_auc_train = roc_auc_score(y_train, y_train_proba)
roc_auc_test = roc_auc_score(y_test, y_test_proba)

print(f"Eğitim Seti ROC-AUC Skoru: {roc_auc_train}")
print(f"Test Seti ROC-AUC Skoru: {roc_auc_test}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=["Churn Yok", "Churn Var"])
cm_display.plot(cmap="viridis")
plt.title("KNN - Confusion Matrix")
plt.show()

# ROC Eğrisi
# Eğitim seti için ROC eğrisi
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
# Test seti için ROC eğrisi
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrilerini çizme
plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, color="blue", label=f"Eğitim ROC (AUC = {roc_auc_train:.2f})")
plt.plot(fpr_test, tpr_test, color="red", label=f"Test ROC (AUC = {roc_auc_test:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Şans Eğrisi")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Eğrisi - Eğitim ve Test Seti (KNN)")
plt.legend(loc="lower right")
plt.grid()
plt.show()



# ### karar ağaçları

# In[200]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
import graphviz
import matplotlib.pyplot as plt

# Karar ağacı modelinin tanımlanması
decision_tree_model = DecisionTreeClassifier(random_state=12345, max_depth=3)  # max_depth sınırlaması görselleştirme için

# Modeli eğitme
decision_tree_model.fit(X_train, y_train)

# Tahmin yapma
y_train_pred = decision_tree_model.predict(X_train)
y_test_pred = decision_tree_model.predict(X_test)

# Performans değerlendirme
print("Eğitim Seti Sınıflandırma Raporu:")
print(classification_report(y_train, y_train_pred))

print("Test Seti Sınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred))

# ROC-AUC Skoru
roc_auc = roc_auc_score(y_test, decision_tree_model.predict_proba(X_test)[:, 1])
print(f"Test Seti ROC-AUC Skoru: {roc_auc}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=["Churn Yok", "Churn Var"])
cm_display.plot(cmap="viridis")
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Karar ağacını görselleştirme
# Graphviz ile
dot_data = export_graphviz(
    decision_tree_model,
    out_file=None,
    feature_names=X.columns,
    class_names=["Churn Yok", "Churn Var"],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)  # Karar ağacını bir PNG dosyasına kaydeder
graph.view()  # Karar ağacını gösterir

# Matplotlib ile alternatif görselleştirme
plt.figure(figsize=(16, 10))
tree.plot_tree(
    decision_tree_model,
    feature_names=X.columns,
    class_names=["Churn Yok", "Churn Var"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree - Matplotlib Görselleştirme")
plt.show()


from sklearn.metrics import roc_curve, auc

# Eğitim seti için ROC eğrisi
fpr_train, tpr_train, _ = roc_curve(y_train, decision_tree_model.predict_proba(X_train)[:, 1])
roc_auc_train = auc(fpr_train, tpr_train)

# Test seti için ROC eğrisi
fpr_test, tpr_test, _ = roc_curve(y_test, decision_tree_model.predict_proba(X_test)[:, 1])
roc_auc_test = auc(fpr_test, tpr_test)

# ROC eğrilerini çizme
plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, color="blue", label=f"Eğitim ROC (AUC = {roc_auc_train:.2f})")
plt.plot(fpr_test, tpr_test, color="red", label=f"Test ROC (AUC = {roc_auc_test:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="RANDOM")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Eğrisi - Eğitim ve Test Seti")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# # lightgbm

# In[202]:


# LightGBM modelinin tanımlanması
lgbm_model = LGBMClassifier(random_state=12345)

# Modeli eğitme
lgbm_model.fit(X_train, y_train)

# Tahmin yapma
y_train_pred = lgbm_model.predict(X_train)
y_test_pred = lgbm_model.predict(X_test)

# Performans değerlendirme
print("Eğitim Seti Sınıflandırma Raporu:")
print(classification_report(y_train, y_train_pred))

print("Test Seti Sınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred))

# ROC-AUC Skoru
roc_auc_train = roc_auc_score(y_train, lgbm_model.predict_proba(X_train)[:, 1])
roc_auc_test = roc_auc_score(y_test, lgbm_model.predict_proba(X_test)[:, 1])
print(f"Eğitim Seti ROC-AUC Skoru: {roc_auc_train}")
print(f"Test Seti ROC-AUC Skoru: {roc_auc_test}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=["Churn Yok", "Churn Var"])
cm_display.plot(cmap="viridis")
plt.title("LightGBM - Confusion Matrix")
plt.show()

# ROC Eğrisi
y_train_proba = lgbm_model.predict_proba(X_train)[:, 1]  # Eğitim kümesi pozitif sınıf olasılıkları
y_test_proba = lgbm_model.predict_proba(X_test)[:, 1]  # Test kümesi pozitif sınıf olasılıkları

# False Positive Rate ve True Positive Rate hesaplama
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC Eğrisini Çizme
plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f"Eğitim ROC Curve (AUC = {roc_auc_train:.4f})")
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f"Test ROC Curve (AUC = {roc_auc_test:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()


from sklearn.metrics import mean_squared_error
import numpy as np

# Test seti için RMSE hesaplama
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_proba))
print(f"Test Seti RMSE: {rmse_test}")

# Eğitim seti için RMSE hesaplama
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_proba))
print(f"Eğitim Seti RMSE: {rmse_train}")


# ### lojistik

# In[201]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Bağımlı ve bağımsız değişkenlerin hazırlanması
#y = df_copy["ayrıldı_mı(churn)"]
#X = df_copy.drop(["ayrıldı_mı(churn)"], axis=1)

# Veriyi eğitim ve test setlerine ayırma
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Lojistik Regresyon modelinin tanımlanması
lr_model = LogisticRegression(random_state=12345, max_iter=1000)

# Modeli eğitme
lr_model.fit(X_train, y_train)

# Tahmin yapma
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Eğitim seti performansı
print("Eğitim Seti Sınıflandırma Raporu:")
print(classification_report(y_train, y_train_pred))

# Test seti performansı
print("Test Seti Sınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred))

# ROC-AUC Skoru
roc_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
print(f"Test Seti ROC-AUC Skoru: {roc_auc}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=["Churn Yok", "Churn Var"])
cm_display.plot(cmap="viridis")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()




# HEM BASE MODELDEN HEM DE LİGHTGBM'E BAKTIĞIMIZDA recall,precision,F1 metrikleri DÜŞÜK PERFORMANS GÖSTERİYOR.AYRICA sıfırı(churn etmeyenler) daha iyi tahmin etmiş.Churn edenleri de iyi tahmin etmek için smote yaptık.

# ## SMOTE: dengesiz dağılan hedef değişkenin dengeli hale getirmek

# In[22]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
# df_copy'den df_copy_smote adında bir kopya oluştur
df_copy_smote = df_copy.copy()
# Bağımlı ve bağımsız değişkenlerin ayrılması
y = df_copy_smote["ayrıldı_mı(churn)"]  # Bağımlı değişken
X = df_copy_smote.drop(["ayrıldı_mı(churn)"], axis=1)  # Bağımsız değişkenler


# In[23]:


from imblearn.over_sampling import SMOTE
from collections import Counter

# SMOTE uygulama
smt = SMOTE(random_state=12345)
X_res, y_res = smt.fit_resample(X, y)  # fit_sample yerine fit_resample kullanılmalı

print(f"Orijinal veri seti dağılımı: {Counter(y)}")
print(f"Resampled veri seti dağılımı: {Counter(y_res)}")


# SMOTE sonrasaı modelin recall,precision,F1 metrikleri artar.
# SMOTE öncesi sınıf dağılımını hesapla
original_class_distribution = y_train.value_counts()

# SMOTE sonrası sınıf dağılımını hesapla
smote_class_distribution = pd.Series(y_res).value_counts()

# Yan yana iki pie grafiği çizdirme
fig, axes = plt.subplots(1, 2, figsize=(9, 5))

# SMOTE öncesi pie grafiği
axes[0].pie(original_class_distribution, 
            labels=['Churn Yok', 'Churn Var'], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=plt.cm.Pastel1.colors)
axes[0].set_title("SMOTE Öncesi Eğitim Verisinde Churn Dağılımı")

# SMOTE sonrası pie grafiği
axes[1].pie(smote_class_distribution, 
            labels=['Churn Yok', 'Churn Var'], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=plt.cm.Pastel1.colors)
axes[1].set_title("SMOTE Sonrası Eğitim Verisinde Churn Dağılımı")

# Grafiklerin gösterimi
plt.tight_layout()
plt.show()
# In[209]:


import matplotlib.pyplot as plt
import seaborn as sns

# SMOTE öncesi scatterplot (df_copy için)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.iloc[:, 0],  # İlk özellik
                y=X.iloc[:, 1],  # İkinci özellik
                hue=y,           # Hedef değişken
                palette=plt.cm.Pastel1.colors)
plt.title("SMOTE Öncesi Eğitim Verisinde Scatterplot")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.legend(title="Churn", labels=["Churn Yok", "Churn Var"])
plt.show()

# SMOTE sonrası scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_smote.iloc[:, 0],  # İlk özellik
                y=X_train_smote.iloc[:, 1],  # İkinci özellik
                hue=y_train_smote,           # Hedef değişken
                palette=plt.cm.Pastel1.colors)
plt.title("SMOTE Sonrası Eğitim Verisinde Scatterplot")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.legend(title="Churn", labels=["Churn Yok", "Churn Var"])
plt.show()

from imblearn.over_sampling import SMOTE
from collections import Counter

# SMOTE uygulama
smt = SMOTE(random_state=12345)
X_res, y_res = smt.fit_resample(X, y)  # fit_sample yerine fit_resample kullanılmalı

print(f"Orijinal veri seti dağılımı: {Counter(y)}")
print(f"Resampled veri seti dağılımı: {Counter(y_res)}")

# ## lightGBM

# In[26]:


from sklearn.metrics import classification_report

#split işlemi
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size=0.20, 
                                                    random_state=12345)

# lgbm model kurulumu
lgbm_model=LGBMClassifier(random_state=12345).fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)

# validasyon hatası, accuracy skoru, confusion matrix
cv_results = cross_val_score(lgbm_model, X_train,y_train, cv = 10, scoring= "accuracy")

print("cross_val_score(train):", cv_results.mean())

cv_results = cross_val_score(lgbm_model, X_test,y_test, cv = 10, scoring= "accuracy")
print("cross_val_score(test):", cv_results.mean())


y_train_pred = lgbm_model.predict(X_train)
print("accuracy_score(train):",accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues');

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Eğitim ve test için tahmin olasılıkları
y_train_proba = lgbm_model.predict_proba(X_train)[:, 1]
y_test_proba = lgbm_model.predict_proba(X_test)[:, 1]

# ROC eğrisi metrikleri
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

from sklearn.metrics import mean_squared_error
import numpy as np

# RMSE hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")



# ### RANDOM FOREST

# In[27]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# RandomForest Modeli
rf_model = RandomForestClassifier(random_state=12345)
rf_model.fit(X_train, y_train)

# Tahminler
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Cross-Validation Accuracy
cv_results_train = cross_val_score(rf_model, X_train, y_train, cv=10, scoring="accuracy")
cv_results_test = cross_val_score(rf_model, X_test, y_test, cv=10, scoring="accuracy")

print(f"Cross-Validation Accuracy (Train): {cv_results_train.mean():.4f}")
print(f"Cross-Validation Accuracy (Test): {cv_results_test.mean():.4f}")

# Accuracy Scores
print(f"Accuracy Score (Train): {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Accuracy Score (Test): {accuracy_score(y_test, y_test_pred):.4f}")

# Classification Report
print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cf_matrix)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.title("Random Forest - Confusion Matrix (Test)")
plt.show()

# ROC Eğrisi
y_train_proba = rf_model.predict_proba(X_train)[:, 1]
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# RMSE Hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")


# #### XGBOOST

# In[28]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Veriyi eğitim ve test setlerine ayırma
#X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=12345)

# XGBoost modelinin kurulumu ve eğitimi
xgb_model = XGBClassifier(random_state=12345, use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# Test tahminleri
y_pred = xgb_model.predict(X_test)

# Cross-validation sonuçları (Eğitim Seti)
cv_results_train = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring="accuracy")
print("cross_val_score(train):", cv_results_train.mean())

# Cross-validation sonuçları (Test Seti)
cv_results_test = cross_val_score(xgb_model, X_test, y_test, cv=10, scoring="accuracy")
print("cross_val_score(test):", cv_results_test.mean())

# Accuracy score
y_train_pred = xgb_model.predict(X_train)
print("accuracy_score(train):", accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

# Eğitim ve test için tahmin olasılıkları
y_train_proba = xgb_model.predict_proba(X_train)[:, 1]
y_test_proba = xgb_model.predict_proba(X_test)[:, 1]

# ROC eğrisi metrikleri
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# RMSE hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")


# ### ADABOOSTING 

# In[221]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# AdaBoost modelinin kurulumu ve eğitimi
ada_model = AdaBoostClassifier(random_state=12345)
ada_model.fit(X_train, y_train)

# Test tahminleri
y_pred = ada_model.predict(X_test)

# Cross-validation sonuçları (Eğitim Seti)
cv_results_train = cross_val_score(ada_model, X_train, y_train, cv=10, scoring="accuracy")
print("cross_val_score(train):", cv_results_train.mean())

# Cross-validation sonuçları (Test Seti)
cv_results_test = cross_val_score(ada_model, X_test, y_test, cv=10, scoring="accuracy")
print("cross_val_score(test):", cv_results_test.mean())

# Accuracy score
y_train_pred = ada_model.predict(X_train)
print("accuracy_score(train):", accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

# ROC eğrisi metrikleri
y_train_proba = ada_model.predict_proba(X_train)[:, 1]
y_test_proba = ada_model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# RMSE hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")


# ## CATBOOST

# In[222]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# CatBoost modelinin kurulumu ve eğitimi
catboost_model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, random_state=12345, cat_features=[], verbose=200)
catboost_model.fit(X_train, y_train)

# Test tahminleri
y_pred = catboost_model.predict(X_test)

# Cross-validation sonuçları (Eğitim Seti)
cv_results_train = cross_val_score(catboost_model, X_train, y_train, cv=10, scoring="accuracy")
print("cross_val_score(train):", cv_results_train.mean())

# Cross-validation sonuçları (Test Seti)
cv_results_test = cross_val_score(catboost_model, X_test, y_test, cv=10, scoring="accuracy")
print("cross_val_score(test):", cv_results_test.mean())

# Accuracy score
y_train_pred = catboost_model.predict(X_train)
print("accuracy_score(train):", accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

# ROC eğrisi metrikleri
y_train_proba = catboost_model.predict_proba(X_train)[:, 1]
y_test_proba = catboost_model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# RMSE hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")


# ### GRADIENT BOOSTING

# In[223]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Gradient Boosting modelinin kurulumu ve eğitimi
gb_model = GradientBoostingClassifier(random_state=12345)
gb_model.fit(X_train, y_train)

# Test tahminleri
y_pred = gb_model.predict(X_test)

# Cross-validation sonuçları (Eğitim Seti)
cv_results_train = cross_val_score(gb_model, X_train, y_train, cv=10, scoring="accuracy")
print("cross_val_score(train):", cv_results_train.mean())

# Cross-validation sonuçları (Test Seti)
cv_results_test = cross_val_score(gb_model, X_test, y_test, cv=10, scoring="accuracy")
print("cross_val_score(test):", cv_results_test.mean())

# Accuracy score
y_train_pred = gb_model.predict(X_train)
print("accuracy_score(train):", accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

# ROC eğrisi metrikleri
y_train_proba = gb_model.predict_proba(X_train)[:, 1]
y_test_proba = gb_model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# RMSE hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")


# ### NAVIE BAYES

# In[224]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Gaussian Naive Bayes modelinin kurulumu ve eğitimi
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# Test tahminleri
y_pred = gnb_model.predict(X_test)

# Cross-validation sonuçları (Eğitim Seti)
cv_results_train = cross_val_score(gnb_model, X_train, y_train, cv=10, scoring="accuracy")
print("cross_val_score(train):", cv_results_train.mean())

# Cross-validation sonuçları (Test Seti)
cv_results_test = cross_val_score(gnb_model, X_test, y_test, cv=10, scoring="accuracy")
print("cross_val_score(test):", cv_results_test.mean())

# Accuracy score
y_train_pred = gnb_model.predict(X_train)
print("accuracy_score(train):", accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

# ROC eğrisi metrikleri
y_train_proba = gnb_model.predict_proba(X_train)[:, 1]
y_test_proba = gnb_model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# RMSE hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")


# ## KNN 

# In[226]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# KNN modelinin kurulumu ve eğitimi
knn_model = KNeighborsClassifier(n_neighbors=5)  # Burada n_neighbors parametresi ayarlanabilir
knn_model.fit(X_train, y_train)

# Test tahminleri
y_pred = knn_model.predict(X_test)

# Cross-validation sonuçları (Eğitim Seti)
cv_results_train = cross_val_score(knn_model, X_train, y_train, cv=10, scoring="accuracy")
print("cross_val_score(train):", cv_results_train.mean())

# Cross-validation sonuçları (Test Seti)
cv_results_test = cross_val_score(knn_model, X_test, y_test, cv=10, scoring="accuracy")
print("cross_val_score(test):", cv_results_test.mean())

# Accuracy score
y_train_pred = knn_model.predict(X_train)
print("accuracy_score(train):", accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

# ROC eğrisi metrikleri
y_train_proba = knn_model.predict_proba(X_train)[:, 1]
y_test_proba = knn_model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# RMSE hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")


# ## KARAR AĞAÇLARI 

# In[228]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from collections import Counter
import graphviz
from sklearn.tree import plot_tree


# Karar Ağacı modelinin kurulumu ve eğitimi
dt_model = DecisionTreeClassifier(random_state=12345, max_depth=3)  # max_depth ile ağacın derinliğini sınırlıyoruz
dt_model.fit(X_train, y_train)

# Test tahminleri
y_pred = dt_model.predict(X_test)

# Cross-validation sonuçları (Eğitim Seti)
cv_results_train = cross_val_score(dt_model, X_train, y_train, cv=10, scoring="accuracy")
print("cross_val_score(train):", cv_results_train.mean())

# Cross-validation sonuçları (Test Seti)
cv_results_test = cross_val_score(dt_model, X_test, y_test, cv=10, scoring="accuracy")
print("cross_val_score(test):", cv_results_test.mean())

# Accuracy score
y_train_pred = dt_model.predict(X_train)
print("accuracy_score(train):", accuracy_score(y_train, y_train_pred))
print("accuracy_score(test):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

# ROC eğrisi metrikleri
y_train_proba = dt_model.predict_proba(X_train)[:, 1]
y_test_proba = dt_model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Eğitim ROC (AUC = {roc_auc_score(y_train, y_train_proba):.2f})", color="blue")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.2f})", color="orange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
plt.xlabel("False Positive Rate (Yanlış Pozitif Oranı)")
plt.ylabel("True Positive Rate (Doğru Pozitif Oranı)")
plt.title("ROC Curve (Eğitim ve Test Setleri)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# RMSE hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Eğitim Seti RMSE: {train_rmse:.4f}")
print(f"Test Seti RMSE: {test_rmse:.4f}")

# Karar ağacını Graphviz ile görselleştirme
dot_data = export_graphviz(
    dt_model,
    out_file=None,
    feature_names=X.columns,
    class_names=["Churn Yok", "Churn Var"],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)  # Karar ağacını bir PNG dosyasına kaydeder
graph.view()  # Karar ağacını gösterir


# In[ ]:




# Test kümesindeki tahminler
y_test_predictions = lgbm_model.predict(X_test)  # Sınıf tahminleri
y_test_probabilities = lgbm_model.predict_proba(X_test)[:, 1]  # Churn olasılıkları (pozitif sınıf)

# Test kümesindeki tahmin sonuçları
test_results = X_test.copy()
test_results['Gerçek'] = y_test.values
test_results['Tahmin'] = y_test_predictions
test_results['Churn Olasılığı'] = y_test_probabilities

test_results.head(10)  # İlk 10 müşteri sonucu

# In[38]:


# Test kümesindeki tahminler
y_test_predictions = lgbm_model.predict(X_test)  # Sınıf tahminleri
y_test_probabilities = lgbm_model.predict_proba(X_test)[:, 1]  # Churn olasılıkları (pozitif sınıf)

# Sadece gerekli sütunları içeren DataFrame oluşturma
test_results = pd.DataFrame({
    'Gerçek': y_test.values,
    'Tahmin': y_test_predictions,
    'Churn Olasılığı': y_test_probabilities
})

# İlk birkaç satırı gösterme
test_results


# In[39]:


# Olasılığı belirli bir eşik değerinden büyük olan müşteriler
threshold = 0.5
churn_customers = test_results[test_results['Churn Olasılığı'] >= threshold]

print(f"Olasılığı {threshold} üzerindeki churn müşteriler:")
churn_customers


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.histplot(y_test_probabilities, bins=20, kde=True, color="blue")
plt.title("Test Kümesindeki Churn Olasılıklarının Dağılımı")
plt.xlabel("Churn Olasılığı")
plt.ylabel("Müşteri Sayısı")
plt.show()


# In[43]:


test_results.to_excel("test_results.xlsx", index=False)
test_results.to_excel(r"C:\Users\HANDENUR\Downloads\test_results.xlsx", index=False)


# In[44]:


test_results.head()


# In[48]:


#!pip install shap

import shap

shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")


# Yaş ve ürün sayısı churn tahmininde en belirleyici faktörlerdir.
# Cinsiyet, ülke, ve aktif üyelik gibi özellikler de önemli katkılarda bulunmaktadır.
# Daha az etkili özellikler olsa da, modelin genel performansı için hepsi bir dereceye kadar değerlidir.
# Elde edilen bu bilgiler, müşteri davranışlarını anlamak ve churn riskini azaltmak için iş stratejileri geliştirmekte kullanılabilir. Örneğin:
# 
# Daha genç müşterilere özel teklifler sunmak.
# Daha az ürün kullanan müşterilere çapraz satış stratejileri geliştirmek.
# Aktif olmayan müşterileri yeniden kazanmaya yönelik kampanyalar düzenlemek.

# ### hangi yaş grubu churn edilmesinde daha etkili

# In[52]:


# Yaşı gruplara ayırma
df['yas_grubu'] = pd.cut(df['yas'], bins=[18, 25, 35, 45, 55, 65, 75], 
                         labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66-75'])

# Her yaş grubu için churn oranını hesaplama
yas_grubu_churn = df.groupby('yas_grubu')['ayrıldı_mı(churn)'].mean()
print(yas_grubu_churn)

# Bar grafiği ile görselleştirme
import matplotlib.pyplot as plt
yas_grubu_churn.plot(kind='bar', color='skyblue', title='Yaş Grubuna Göre Churn Oranı')
plt.xlabel('Yaş Grubu')
plt.ylabel('Churn Oranı')
plt.show()


# Şirketi en çok  terk eden müşterilerin yaş grubu 46-55 ve 56-65 aralığı imiş.

# ### karar ağacına göre bakalım

# In[55]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Basit bir karar ağacı modeli
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X_train[['yas']], y_train)

# Karar ağacını görselleştirme
plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=['yas'], class_names=['Kalmış', 'Ayrılmış'], filled=True)
plt.show()


# In[61]:


# Yaş ve diğer özelliklere göre churn oranı
segment_churn = df.groupby(['yas_grubu', 'aktif_uye_mi'])['ayrıldı_mı(churn)'].mean()
print(segment_churn)


# Yaş aralığı 56-65 olan ve aktif üye olmayanlar şirketi en çok terk eden müşteriler.

# In[ ]:




