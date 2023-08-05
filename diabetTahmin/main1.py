import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Veri kümesini yükleyelim (örneğin, diabetes veri kümesi kullanılacak)
dosya = pd.read_csv("data.csv")
xDegerleri = dosya.drop("Outcome", axis=1)
yDegerleri = dosya["Outcome"]

# Veri kümesini eğitim ve test verilerine bölelim
X_train, X_test, y_train, y_test = train_test_split(xDegerleri, yDegerleri, test_size=0.2, random_state=42)

# Hyperparametre değerlerini belirleyelim
n_estimators_values_rf = [50, 100, 150, 200]
max_depth_values_rf = [None, 5, 10, 15]  # None, sınırsız derinlik anlamına gelir
n_estimators_values_ab = [50, 100, 150, 200]

# Çizgi grafikleri için verileri saklayalım
rf_results = []
ab_results = []

for n_estimators_rf in n_estimators_values_rf:
    for max_depth_rf in max_depth_values_rf:
        # RandomForestClassifier'ı belirtilen hiperparametrelerle eğitelim
        rf = RandomForestClassifier(n_estimators=n_estimators_rf, max_depth=max_depth_rf, random_state=42)
        rf.fit(X_train, y_train)

        # Test verileri üzerinde tahmin yapalım
        y_pred_rf = rf.predict(X_test)

        # Model doğruluğunu değerlendirelim
        accuracy_rf = accuracy_score(y_test, y_pred_rf)

        # Sonuçları rf_results listesine ekleyelim
        rf_results.append((n_estimators_rf, max_depth_rf, accuracy_rf))

    # AdaBoostClassifier'ı belirtilen hiperparametrelerle eğitelim
    ab = AdaBoostClassifier(n_estimators=n_estimators_rf, learning_rate=0.1, random_state=42)
    ab.fit(X_train, y_train)

    # Test verileri üzerinde tahmin yapalım
    y_pred_ab = ab.predict(X_test)

    # Model doğruluğunu değerlendirelim
    accuracy_ab = accuracy_score(y_test, y_pred_ab)

    # Sonuçları ab_results listesine ekleyelim
    ab_results.append((n_estimators_rf, accuracy_ab))

# Grafik çizimi için verileri ayrıştıralım
rf_n_estimators_list = [res[0] for res in rf_results]
rf_max_depth_list = [res[1] for res in rf_results]
rf_accuracy_list = [res[2] for res in rf_results]

ab_n_estimators_list = [res[0] for res in ab_results]
ab_accuracy_list = [res[1] for res in ab_results]

# Çizgi grafiklerini oluşturalım
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for max_depth_rf in max_depth_values_rf:
    rf_max_depth_acc = [rf_accuracy_list[i] for i in range(len(rf_accuracy_list)) if rf_max_depth_list[i] == max_depth_rf]
    plt.plot(rf_n_estimators_list[:len(rf_max_depth_acc)], rf_max_depth_acc,
             marker='o', label=f'max_depth={max_depth_rf}')
plt.xlabel('n_estimators')
plt.ylabel('Doğruluk')
plt.title('RF için max_depth değişimi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rf_n_estimators_list, rf_accuracy_list, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('Doğruluk')
plt.title('RF için n_estimators değişimi')

plt.tight_layout()
plt.savefig("sonuc.png")
plt.show()
