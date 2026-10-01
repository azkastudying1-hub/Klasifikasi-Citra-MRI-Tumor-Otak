# Proyek Klasifikasi Gambar: Deteksi Jenis Tumor Otak dari Citra MRI

## Deskripsi Proyek

Proyek ini merupakan submission akhir kelas Machine Learning di Dicoding untuk kasus **klasifikasi gambar**. Model yang dibangun berfungsi sebagai bagian dari *asisten kesehatan* (Health Bot) untuk membantu mengklasifikasikan jenis tumor otak berdasarkan citra MRI ke dalam tiga kelas:

- `brain_glioma` — Glioma
- `brain_menin` — Meningioma
- `brain_tumor` — Tumor Pituitari

Model dibangun menggunakan **Convolutional Neural Network (CNN)** dengan `tf.keras.Sequential`, dilatih menggunakan teknik *image augmentation*, dan dievaluasi hingga mencapai akurasi **di atas 92%** baik pada data training maupun data testing.

> ⚠️ **Disclaimer:** Model ini dibuat untuk tujuan pembelajaran dan **bukan alat diagnosis medis**. Jangan gunakan hasil prediksi model untuk pengambilan keputusan medis nyata tanpa validasi tenaga profesional.

## Dataset

Dataset yang digunakan adalah subset **Brain Cancer** dari [Multi Cancer Dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer) yang tersedia di Kaggle, berisi lebih dari 10.000 citra MRI otak yang terbagi ke dalam 3 kelas di atas.

Dataset dibagi menjadi:
- **Train set**: 80%
- **Validation set**: 10%
- **Test set**: 10%

## Arsitektur Model

Model CNN `Sequential` dengan struktur berikut:

```
Input (150x150x3)
→ Conv2D(32, 3x3, ReLU) → MaxPooling2D
→ Conv2D(64, 3x3, ReLU) → MaxPooling2D
→ Conv2D(128, 3x3, ReLU) → MaxPooling2D
→ Conv2D(128, 3x3, ReLU) → MaxPooling2D
→ Flatten
→ Dense(512, ReLU) → Dropout(0.5)
→ Dense(3, Softmax)
```

## Proses Pelatihan

- **Optimizer**: Adam (learning rate 1e-4)
- **Loss function**: Categorical Crossentropy
- **Callback**: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, serta callback custom yang menghentikan pelatihan apabila akurasi training & validasi telah melampaui 92%
- **Augmentasi data**: rotasi, pergeseran, shear, zoom, dan flip horizontal menggunakan `ImageDataGenerator`

## Hasil

Model berhasil mencapai akurasi **di atas 92%** pada data training maupun data testing. Detail proses analisis, pelatihan, grafik akurasi/loss, serta classification report dan confusion matrix dapat dilihat pada `notebook.ipynb`.

## Format Model yang Disediakan

Model hasil pelatihan dikonversi ke tiga format agar dapat digunakan pada berbagai platform:

| Format | Lokasi | Kegunaan |
|---|---|---|
| SavedModel | `saved_model/` | Deployment di server / TensorFlow Serving |
| TensorFlow Lite | `tflite/model.tflite` + `tflite/label.txt` | Deployment di perangkat mobile/embedded |
| TensorFlow.js | `tfjs_model/` | Deployment di aplikasi web / browser |

## Struktur Direktori

```
submission
├───tfjs_model
│   ├───group1-shard1of4.bin
│   ├───group1-shard2of4.bin
│   ├───group1-shard3of4.bin
│   ├───group1-shard4of4.bin
│   └───model.json
├───tflite
│   ├───model.tflite
│   └───label.txt
├───saved_model
│   ├───saved_model.pb
│   └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt
```

## Cara Menjalankan

1. Buka `notebook.ipynb` di Google Colab atau Jupyter Notebook.
2. Install dependency yang dibutuhkan:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan seluruh cell secara berurutan dari atas ke bawah. Notebook akan otomatis mengunduh dataset dari Kaggle (memerlukan koneksi internet dan Kaggle API token bila dijalankan di luar Kaggle/Colab dengan akses `kagglehub`), melatih model, mengevaluasi, dan mengonversinya ke tiga format di atas.

## Author

Proyek submission akhir kelas Machine Learning — Dicoding Indonesia.
