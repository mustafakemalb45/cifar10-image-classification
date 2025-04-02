import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# Sınıf isimleri
class_names = ['Uçak', 'Araba', 'Kuş', 'Kedi', 'Geyik', 'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

# Veriyi yükle
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize et
x_train, x_test = x_train / 255.0, x_test / 255.0

# Modeli tanımla
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Eğit
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Test doğruluğu
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test doğruluğu: {test_acc:.4f}")

# Rastgele bir görüntü göster ve tahmini yaz
index = np.random.randint(0, len(x_test))
img = x_test[index]
true_label = y_test[index][0]
pred = model.predict(np.expand_dims(img, axis=0))
pred_label = np.argmax(pred)

plt.imshow(img)
plt.title(f"Tahmin: {class_names[pred_label]} / Gerçek: {class_names[true_label]}")
plt.axis('off')
plt.show()
