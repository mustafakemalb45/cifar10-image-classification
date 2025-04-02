# Aktivasyon fonksiyonu (Sigmoid)
def sigmoid(x):
    return 1 / (1 + 2.71828 ** (-x))

# Giriş verileri
x1 = 0.5
x2 = 0.8

# Ağırlıklar ve bias
w1 = 0.944
w2 = 0.1
bias = -0.5555

# Nöronun toplam girdisi
z = x1 * w1 + x2 * w2 + bias

# Aktivasyon fonksiyonu uygulandıktan sonraki çıktı
y = sigmoid(z)

# Sonuç
print("Nöronun çıktısı:", y)
