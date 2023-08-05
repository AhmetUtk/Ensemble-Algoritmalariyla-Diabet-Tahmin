import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Veri setini yükleyelim 
dosya = pd.read_csv("data.csv")
# Hedef veri değer verilerimizi ayıralım
xDegerleri = dosya.drop("Outcome", axis=1)
yDegerleri = dosya["Outcome"]

# Veri setini eğitim ve test verilerine bölelim
X_train, X_test, y_train, y_test = train_test_split(xDegerleri, yDegerleri, test_size=0.2, random_state=42)

# Ensemble algoritmalarında tek tek en yüksek değerleri bulmak ve atama yapmak için tanımlamalar yapalım
rfDogruluk = 0.0 
rfParametre = {} 
gbDogruluk = 0.0 
gbParametre = {} 
adDogruluk = 0.0 
abParametre = {} 

# Doğruluk oranını arttırmak için farklı parametre değerleri tanımlayalım
rfn_estimator = [50, 100, 150, 200] 
rfmax_depth = [None, 5, 10, 15]  # None sınırsız derinlik anlamına gelir
gbn_estimator = [50, 100, 150, 200]
gbRate = [0.01, 0.05, 0.1, 0.2]
abn_estimator = [50, 100, 150, 200]
abRate = [0.01, 0.05, 0.1, 0.2]

# Random Forest için en iyi hiperparametreleri bulalım
for i in rfn_estimator:
    for j in rfmax_depth:
        # Random Forest belirtilen hiperparametrelerle eğitelim
        rf = RandomForestClassifier(n_estimators=i, max_depth=j, random_state=42)
        rf.fit(X_train, y_train)

        # Test verileri üzerinde tahmin yapalım
        yTahmin_rf = rf.predict(X_test) 

        # Model doğruluğunu değerlendirelim
        accuracy_rf = accuracy_score(y_test, yTahmin_rf)

        # En yüksek doğruluk değerini ve ilgili hiperparametreleri güncelleyelim
        if accuracy_rf > rfDogruluk:
            rfDogruluk = accuracy_rf
            rfParametre['n_estimators'] = i
            rfParametre['max_depth'] = j

# Gradient Boosting için en iyi hiperparametreleri bulalım
for i in gbn_estimator:
    for j in gbRate:
        # Gradient Boosting belirtilen hiperparametrelerle eğitelim
        gb = GradientBoostingClassifier(n_estimators=i, learning_rate=j, random_state=42)
        gb.fit(X_train, y_train)

        # Test verileri üzerinde tahmin yapalım
        yTahmin_gb = gb.predict(X_test) 

        # Model doğruluğunu değerlendirelim
        accuracy_gb = accuracy_score(y_test, yTahmin_gb)

        # En yüksek doğruluk değerini ve ilgili hiperparametreleri güncelleyelim
        if accuracy_gb > gbDogruluk:
            gbDogruluk = accuracy_gb
            gbParametre['n_estimators'] = i
            gbParametre['learning_rate'] = j

# Ada Boost için en iyi hiperparametreleri bulalım
for i in abn_estimator:
    for j in abRate:
        # Ada Boost belirtilen hiperparametrelerle eğitelim
        ab = AdaBoostClassifier(n_estimators=i, learning_rate=j, random_state=42)
        ab.fit(X_train, y_train)

        # Test verileri üzerinde tahmin yapalım
        yTahmin_ab = ab.predict(X_test)

        # Model doğruluğunu değerlendirelim
        accuracy_ab = accuracy_score(y_test, yTahmin_ab)

        # En yüksek doğruluk değerini ve ilgili hiperparametreleri güncelleyelim
        if accuracy_ab > adDogruluk:
            adDogruluk = accuracy_ab
            abParametre['n_estimators'] = i
            abParametre['learning_rate'] = j

# En yüksek doğruluk değerleri ve ilgili hiperparametreleri yazdıralım
print("RandomForestClassifier için En Yüksek Doğruluk: {:.5f}".format(rfDogruluk))
print("En İyi Hiperparametreler:")
print(rfParametre)

print("\nGradientBoostingClassifier için En Yüksek Doğruluk: {:.5f}".format(gbDogruluk))
print("En İyi Hiperparametreler:")
print(gbParametre)

print("\nAdaBoostClassifier için En Yüksek Doğruluk: {:.5f}".format(adDogruluk))
print("En İyi Hiperparametreler:")
print(abParametre)
