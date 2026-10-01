# 🧠 Klasifikasi Citra MRI Tumor Otak (Brain MRI Tumor Classification)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Deep Learning](https://img.shields.io/badge/Field-Deep%20Learning-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

Sistem otomatisasi berbasis **Deep Learning** dan **Convolutional Neural Network (CNN)** untuk mengklasifikasikan citra *Magnetic Resonance Imaging* (MRI) otak. Proyek ini dikembangkan untuk membantu mendeteksi dan mengkategorikan jenis tumor otak secara presisi dari citra medis digital.

---

## 📌 Ringkasan Proyek

Pendeteksian dini tumor otak melalui citra MRI memainkan peran krusial dalam perencanaan tindakan medis. Proyek ini memanfaatkan model *Deep Learning* yang dilatih menggunakan arsitektur pemrosesan citra untuk mengenali pola abnormalitas pada jaringan otak. Model yang dikembangkan diekspor ke dalam format standar **TensorFlow SavedModel** untuk mempermudah integrasi dan *deployment* pada sistem medis lanjutan atau aplikasi AI edge.

---

## ✨ Fitur Utama

* **Preprocessing Citra Medis**: Normalisasi citra, perbaikan kontras, dan *resizing* otomatis untuk memasok data berkualitas tinggi ke model.
* **Klasifikasi Berbasis Deep Learning**: Memanfaatkan arsitektur Convolutional Neural Network (CNN) untuk mengekstrak fitur spasial kompleks dari citra MRI.
* **Artefak Model Siap Pakai**: Model disimpan dalam format `SavedModel` (`.pb`), memungkinkan pemanggilan ulang untuk inferensi tanpa perlu pelatihan ulang (*retraining*).
* **Ekosistem Modular**: Struktur kode yang rapi dan mudah diintegrasikan dengan API web (seperti Flask/FastAPI) atau aplikasi *deployment* lainnya.

---

## 📂 Struktur Direktori

```text
├── saved_model/              # Artefak model TensorFlow yang telah dilatih
│   ├── assets/               # File tambahan pendukung model
│   ├── variables/            # Bobot (weights) dan parameter teroptimasi
│   └── saved_model.pb        # Graf arsitektur TensorFlow
├── notebook/                 # Notebook eksperimen dan pelatihan model
├── submission/               # Modul pemrosesan dan kode inferensi
├── requirements.txt          # Daftar dependensi library Python
└── README.md                 # Dokumentasi proyek
```

---

## 📋 Prasyarat Sistem

Sebelum memulai, pastikan perangkat Anda telah memenuhi spesifikasi minimum berikut:

* **Python**: Versi `3.8` hingga `3.11`
* **Git**: Terinstal di sistem Anda
* **RAM**: Minimal 8 GB *(Disarankan menggunakan GPU/CUDA jika ingin melakukan training ulang)*

---

## 📥 Langkah Instalasi

### Langkah 1: Kloning Repositori
Buka terminal atau Command Prompt (`cmd`) Anda, lalu jalankan perintah berikut:
```bash
git clone https://github.com
cd Klasifikasi-Citra-MRI-Tumor-Otak
```

### Langkah 2: Buat & Aktifkan Virtual Environment
Penggunaan *virtual environment* sangat disarankan untuk menghindari bentrok versi pustaka (*library*).

* **Windows (CMD / PowerShell):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Langkah 3: Perbarui PIP & Instal Dependensi
Jalankan perintah berikut untuk memperbarui pengelola paket dan menginstal seluruh pustaka yang dibutuhkan:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Cara Penggunaan

### A. Inferensi Menggunakan SavedModel
Anda dapat langsung menguji model terhadap citra MRI baru menggunakan skrip Python sederhana di bawah ini:

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Tentukan path ke folder SavedModel dan file gambar
MODEL_PATH = 'saved_model'
IMAGE_PATH = 'sample_mri.jpg'  # Ubah sesuai lokasi citra tes Anda

# 2. Muat model TensorFlow
print("Memuat model...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures['serving_default']

# 3. Preprocessing Citra MRI (Ubah ukuran sesuai spesifikasi input model, misal 150x150)
def preprocess_image(img_path, target_size=(150, 150)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalisasi [0, 1]
    img_array = np.expand_dims(img_array, axis=0)        # Tambahkan dimensi batch
    return tf.constant(img_array)

# 4. Jalankan Prediksi
input_tensor = preprocess_image(IMAGE_PATH)
output = infer(input_tensor)

# 5. Tampilkan Hasil Prediksi
# Sesuaikan key output (biasanya 'dense' atau nama layer output terakhir)
predictions = list(output.values())[0].numpy()
predicted_class = np.argmax(predictions[0])

# Label Kelas (Sesuaikan dengan kelas pada dataset Anda)
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

print(f"Hasil Diagnosa : {class_labels[predicted_class]}")
print(f"Tingkat Kepercayaan : {predictions[0][predicted_class] * 100:.2f}%")
```

### B. Menjalankan Melalui Jupyter Notebook
Jika Anda ingin melakukan eksperimen atau melihat proses analisis data secara visual, jalankan perintah berikut:
```bash
pip install jupyter
jupyter notebook
```
Buka file `.ipynb` yang tersedia di dalam folder repositori melalui antarmuka Jupyter yang terbuka secara otomatis di browser Anda.

---

## ❓ Pemecahan Masalah (Troubleshooting)

Jika Anda menemui kendala saat instalasi atau penggunaan, berikut adalah beberapa solusi yang bisa dicoba:

| Masalah | Solusi |
| :--- | :--- |
| `ModuleNotFoundError` | Pastikan *virtual environment* sudah aktif, lalu jalankan kembali perintah `pip install -r requirements.txt`. |
| `MemoryError` saat memuat model | Kurangi ukuran *batch* (jika melakukan *training*) atau alihkan penggunaan ke CPU jika alokasi VRAM GPU tidak mencukupi. |
| Ukuran file `saved_model` terlalu besar untuk `git push` | Pastikan folder `saved_model/` telah ditambahkan ke dalam file `.gitignore` jika ukurannya melebihi 100 MB. |
