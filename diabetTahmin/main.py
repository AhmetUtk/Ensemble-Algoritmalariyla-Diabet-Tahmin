import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Veri setini oku
dosya = pd.read_csv("data.csv")

# "Outcome" sütununu hedef değişken olarak ayır
yDeger = dosya["Outcome"]
xDeger = dosya.drop("Outcome", axis=1)

# Veriyi eğitim ve test kümelerine böle (random_state belirleyerek tekrarlanabilir sonuç edebilirsiniz)
x_train, x_test, y_train, y_test = train_test_split(xDeger, yDeger, test_size=0.25, random_state=42)

# Verileri ölçeklendir
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Yapay sinir ağı modelini oluştur
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Modeli derle
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Modeli eğit ve eğitim sürecindeki metrik değerleri history değişkenine kaydet
history = model.fit(x_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(x_test_scaled, y_test))

# Eğitim ve doğruluk değerlerini çizdir
plt.plot(history.history['accuracy'], label='Eğitim Doğruluk')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluk')
plt.plot(history.history['loss'], label='Eğitim Kayıp')
plt.plot(history.history['val_loss'], label='Doğrulama Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Değer')
plt.legend()
plt.savefig("projeGrafik.png")
plt.show()

# Eğitim ve test verileri üzerindeki doğruluk değerlerini yazdır
train_accuracy = history.history['accuracy'][-1]
test_accuracy = history.history['val_accuracy'][-1]
print("Eğitim verileri üzerinde doğruluk:", train_accuracy)
print("Test verileri üzerinde doğruluk:", test_accuracy)
