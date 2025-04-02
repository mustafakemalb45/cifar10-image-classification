# CIFAR-10 Görüntü Sınıflandırma Projesi 🧠📦

Bu proje, TensorFlow ve Keras kullanılarak CIFAR-10 veri seti üzerinde temel bir görüntü sınıflandırma modeli eğitir. Eğitim sonucunda model `.h5` formatında kaydedilir.

---

## 🚀 Kullanılan Teknolojiler

- Python 3.10
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## 📊 Model Mimarisi

- 2 x Conv2D + MaxPooling
- Flatten
- Dense (ReLU)
- Dense (Softmax - 10 sınıf)

Model, 10 epoch boyunca eğitilir. Test doğruluğu ortalama %70–75 civarındadır.

---

## ⚙️ Kurulum ve Kullanım

### 1. Sanal Ortam Kurulumu

```bash
python -m venv dl_env
.\dl_env\Scripts\activate
pip install -r requirements.txt

