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


---

## ☢️ Geant4 TestEm13 Kurulum Paketi (Güncel)

Bu repoda Cs-137 gamma ölçümü için TestEm13 uyumlu, daha gerçekçi bir paket bulunur:

- `geant4/testem13_setup/DetectorConstruction.hh/.cc`
- `geant4/testem13_setup/PrimaryGeneratorAction.hh/.cc`
- `geant4/testem13_setup/ActionInitialization.hh/.cc`
- `geant4/testem13_setup/run_cs137_gauss.mac`
- `geant4/testem13_setup/README_TR.md`

Not: Cs-137 enerjisi sabit değil, gaussian dağılımdan örneklenir (661.657 keV merkezli).
