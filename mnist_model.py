import sys
print("Kullanılan Python yolu:", sys.executable)

import tensorflow as tf
print("TensorFlow versiyonu:", tf.__version__)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Veri setini yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Veriyi normalleştir
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Etiketleri kategorik hale getir
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 4. Modeli oluştur
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 sınıf var: 0–9 arası
])

# 5. Derle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Eğit
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 7. Test et
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test doğruluğu:", test_acc)
