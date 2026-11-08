# IDENTIFIKASI DATASET MACHINE LEARNING

Dokumentasi ini berisi identifikasi sembilan dataset yang dipilih untuk pembelajaran supervised learning. Dataset dikelompokkan berdasarkan jenis target, yaitu Binary Classification, Multiclass Classification, dan Multilabel Classification.

## 1. BINARY CLASSIFICATION

Binary Classification merupakan metode klasifikasi yang bertujuan mengelompokkan setiap data ke dalam salah satu dari dua kategori. Pada kategori ini, ketiga dataset menggunakan target yang menunjukkan dua kondisi berbeda.

### 1.1 Breast Cancer Wisconsin (Diagnostic) Data Set

- **Sumber Dataset:** [Kaggle – Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Jumlah Data:** 569 observasi dengan 32 kolom pada file CSV. Terdapat 30 variabel pengukuran, satu kolom identitas, dan satu kolom diagnosis.
- **Variabel Target:** `diagnosis`, yang menunjukkan hasil diagnosis tumor berdasarkan dua kategori, yaitu B (Benign) dan M (Malignant).
- **Jenis Permasalahan:** Binary Classification. Model bertugas menentukan apakah karakteristik tumor termasuk kategori jinak atau ganas. Kedua kategori tersebut merupakan kelas yang saling terpisah.
- **Distribusi Kelas:** 357 data termasuk Benign (62,7%), sedangkan 212 data termasuk Malignant (37,3%).
- **Variabel Prediktor:** Terdiri dari pengukuran karakteristik inti sel, meliputi `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry`, dan `fractal dimension`. Setiap karakteristik memiliki nilai mean, standard error, dan worst.
- **Karakteristik Dataset:** Data berbentuk numerik sehingga dapat digunakan untuk pengujian algoritma klasifikasi seperti Logistic Regression, KNN, SVM, Decision Tree, dan Random Forest.
- **Hal yang Perlu Diperhatikan:** Kolom `id` tidak digunakan sebagai variabel prediktor karena hanya berfungsi sebagai identitas sampel. Jika terdapat kolom `Unnamed: 32` pada file Kaggle, kolom tersebut perlu diperiksa karena bukan fitur penelitian dan dapat dihapus apabila seluruh nilainya kosong. Dataset ini tidak memiliki missing value pada atribut aslinya.

### 1.2 Heart Disease Classification (UCI)

- **Sumber Dataset:** [Kaggle – Heart Disease UCI](https://www.kaggle.com/competitions/heart-disease-uci)
- **Asal Data:** Cleveland Heart Disease Dataset yang berasal dari penelitian mengenai kondisi penyakit jantung.
- **Struktur Data:** Memuat 14 atribut, termasuk variabel diagnosis. Data kompetisi disediakan dalam file pelatihan dan pengujian.
- **Variabel Target:** `num`, yang menunjukkan tingkat diagnosis penyakit jantung. Nilai 0 menunjukkan tidak terdapat penyakit jantung, sedangkan nilai 1, 2, 3, dan 4 menunjukkan adanya penyakit jantung.
- **Jenis Permasalahan:** Binary Classification setelah dilakukan pengelompokan nilai target. Kategori 0 dipertahankan sebagai kelas tidak memiliki penyakit jantung, sementara nilai 1–4 digabung menjadi kelas memiliki penyakit jantung.
- **Kategori Target:**
  - 0 = Tidak terindikasi penyakit jantung.
  - 1 = Terindikasi penyakit jantung.
- **Variabel Prediktor:** Meliputi `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, dan `thal`.
- **Karakteristik Dataset:** Menggabungkan data numerik dan kategorikal yang menggambarkan kondisi kesehatan pasien, seperti usia, tekanan darah, kolesterol, jenis nyeri dada, dan hasil pemeriksaan jantung.
- **Hal yang Perlu Diperhatikan:** Nilai target perlu dikonversi terlebih dahulu agar menjadi dua kategori. Dataset juga memiliki beberapa nilai kosong sehingga perlu dilakukan pemeriksaan dan penanganan missing value sebelum pemodelan. Pembagian data pelatihan dan pengujian harus mengikuti struktur file kompetisi.

### 1.3 Credit Card Fraud Detection

- **Sumber Dataset:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Periode Data:** Transaksi kartu kredit yang dilakukan oleh pemegang kartu di Eropa selama dua hari pada September 2013.
- **Jumlah Data:** 284.807 transaksi dengan 31 kolom.
- **Variabel Target:** `Class`, yang digunakan untuk membedakan transaksi normal dan transaksi yang terindikasi penipuan.
- **Jenis Permasalahan:** Binary Classification karena setiap transaksi hanya memiliki satu dari dua kemungkinan label, yaitu transaksi sah atau transaksi penipuan.
- **Distribusi Kelas:**
  - Class 0 (Normal): 284.315 transaksi atau sekitar 99,83%.
  - Class 1 (Fraud): 492 transaksi atau sekitar 0,17%.
- **Variabel Prediktor:** Terdiri dari `Time`, `Amount`, serta 28 variabel hasil transformasi Principal Component Analysis (PCA), yaitu `V1` sampai `V28`.
- **Karakteristik Dataset:** Seluruh fitur prediktor berbentuk numerik. Sebagian besar variabel telah melalui transformasi PCA untuk menjaga kerahasiaan informasi transaksi.
- **Hal yang Perlu Diperhatikan:** Dataset memiliki ketidakseimbangan kelas yang sangat tinggi. Oleh karena itu, evaluasi model tidak cukup hanya menggunakan accuracy. Precision, recall, F1-score, confusion matrix, dan PR-AUC perlu dipertimbangkan. Penanganan class imbalance juga dapat dilakukan menggunakan class weighting atau teknik resampling pada data training.

---

## 2. MULTICLASS CLASSIFICATION

Multiclass Classification merupakan metode klasifikasi yang digunakan ketika suatu permasalahan memiliki lebih dari dua kategori target. Setiap observasi hanya ditetapkan pada satu kelas dari sejumlah kelas yang tersedia.

### 2.1 Dry Bean Classification

- **Sumber Dataset:** [Kaggle – Dry Bean Classification](https://www.kaggle.com/code/deanbatur/dry-bean-classification)
- **Referensi Dataset Asli:** UCI Machine Learning Repository – Dry Bean Dataset.
- **Jumlah Data:** 13.611 sampel biji kacang kering dengan 16 variabel prediktor.
- **Variabel Target:** `Class`, yang menunjukkan jenis atau varietas kacang kering berdasarkan karakteristik bentuk dan ukurannya.
- **Jenis Permasalahan:** Multiclass Classification karena terdapat tujuh kategori varietas kacang dan setiap sampel hanya memiliki satu label kelas.
- **Kategori Target:** Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, dan Sira.
- **Variabel Prediktor:** Meliputi `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, serta empat variabel `ShapeFactor`.
- **Karakteristik Dataset:** Data numerik diperoleh melalui pengukuran karakteristik fisik biji kacang yang sebelumnya difoto menggunakan kamera beresolusi tinggi. Hasil citra kemudian diproses untuk menghasilkan fitur pengukuran.
- **Hal yang Perlu Diperhatikan:** Tautan Kaggle yang dipilih merupakan notebook pengolahan dan klasifikasi, bukan halaman dataset tersendiri. Dataset sumbernya perlu dipastikan menggunakan file Dry Bean asli. Kolom `Class` berfungsi sebagai target, sedangkan 16 atribut lainnya digunakan sebagai prediktor. Dataset asli tidak memiliki missing value.

### 2.2 Rice Seed Image Dataset from Pakistan

- **Sumber Dataset:** [Kaggle – Rice Seed Image Dataset from Pakistan](https://www.kaggle.com/datasets/ishaqedu/rice-seed-image-dataset-from-pakistan)
- **Jumlah Data:** 10.000 gambar biji beras yang terdiri dari 8.000 gambar untuk training dan 2.000 gambar untuk testing.
- **Objek Pengamatan:** Citra biji beras yang dikumpulkan menggunakan kamera smartphone dengan latar belakang kertas berwarna hitam dan pencahayaan tambahan.
- **Variabel Target:** Label varietas beras yang menunjukkan jenis beras pada masing-masing gambar.
- **Jenis Permasalahan:** Multiclass Classification karena setiap gambar dikategorikan ke dalam satu dari tujuh varietas beras.
- **Kategori Target:** Kachi, Seela, Sufaid, Super, 1508, Ari, dan Kachi Kainat.
- **Karakteristik Dataset:** Data berupa citra digital (JPG) yang dapat digunakan sebagai input model pengenalan pola visual. Setiap kategori memiliki jumlah gambar yang sama.
- **Hal yang Perlu Diperhatikan:** Karena data berbentuk gambar, proses pengolahan mencakup penyesuaian ukuran citra, normalisasi pixel, dan pembagian data. Dataset ini lebih sesuai untuk algoritma berbasis pengolahan citra dan Deep Learning, khususnya Convolutional Neural Network (CNN). Label varietas harus tetap dipertahankan sebagai satu kelas untuk setiap gambar.

### 2.3 CI - Letter Recognition

- **Sumber Dataset:** [Kaggle – CI Letter Recognition](https://www.kaggle.com/competitions/ci-letter-recognition)
- **Asal Dataset:** Letter Recognition Dataset dari UCI Machine Learning Repository.
- **Jumlah Data:** 20.000 sampel karakter dengan 16 atribut numerik sebagai fitur prediktor.
- **Variabel Target:** `letter`, yang menunjukkan huruf kapital dalam alfabet bahasa Inggris.
- **Jenis Permasalahan:** Multiclass Classification karena model harus membedakan 26 kategori huruf, mulai dari A sampai Z.
- **Kategori Target:** A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, dan Z.
- **Variabel Prediktor:** Terdiri dari 16 atribut, yaitu `x-box`, `y-box`, `width`, `high`, `onpix`, `x-bar`, `y-bar`, `x2bar`, `y2bar`, `xybar`, `x2ybr`, `xy2br`, `x-ege`, `xegvy`, `y-ege`, dan `yegvx`.
- **Karakteristik Dataset:** Setiap sampel merepresentasikan karakter huruf yang berasal dari tampilan pixel hitam-putih. Informasi visual tersebut telah diubah menjadi atribut numerik yang menggambarkan posisi, ukuran, kepadatan pixel, serta karakteristik tepi huruf.
- **Hal yang Perlu Diperhatikan:** Kolom `letter` menjadi target klasifikasi, sedangkan 16 atribut lainnya menjadi fitur masukan. File kompetisi terdiri atas data training, testing, dan contoh submission. Kompetisi mengharuskan peserta menyetujui aturan sebelum mengakses data.

---

## 3. MULTILABEL CLASSIFICATION

Multilabel Classification merupakan pendekatan klasifikasi yang memungkinkan satu observasi memiliki lebih dari satu label secara bersamaan. Berbeda dari multiclass, setiap data tidak harus masuk ke satu kategori saja karena beberapa kategori dapat muncul pada objek yang sama.

### 3.1 Tomato-Village Dataset

- **Sumber Dataset:** [Kaggle – Tomato-Village](https://www.kaggle.com/datasets/mamtag/tomato-village)
- **Referensi Penelitian:** Gehlot, Saxena, dan Gandhi (2023), berjudul *“Tomato-Village”: A Dataset for End-to-End Tomato Disease Detection in a Real-World Environment*.
- **Objek Pengamatan:** Gambar daun tanaman tomat yang dikumpulkan langsung dari lingkungan perkebunan di Rajasthan, India.
- **Jumlah Data:** Pada penelitian awal, terdapat 3.231 gambar yang digunakan dalam pengembangan varian multilabel. Versi Kaggle dan publikasi lanjutan melaporkan pembagian data yang berbeda, sehingga jumlah file aktual perlu dicocokkan dengan versi yang diunduh.
- **Variabel Target:** Kumpulan label yang menggambarkan kondisi kesehatan dan penyakit yang ditemukan pada daun tomat.
- **Jenis Permasalahan:** Multilabel Classification karena satu gambar dapat memperlihatkan lebih dari satu jenis penyakit atau kondisi tanaman sekaligus.
- **Kategori Label:** Early Blight, Healthy, Late Blight, Leaf Miner, Magnesium Deficiency, Nitrogen Deficiency, Potassium Deficiency, dan Spotted Wilt Virus.
- **Representasi Target:** Setiap kategori penyakit dapat direpresentasikan menggunakan kolom binary 0 dan 1. Nilai 1 menunjukkan keberadaan kategori tersebut pada gambar, sedangkan 0 menunjukkan kategori tidak ditemukan.
- **Karakteristik Dataset:** Menggunakan citra daun tomat dengan kondisi pengambilan gambar di lingkungan nyata, sehingga variasi latar belakang dan kondisi pencahayaan lebih beragam.
- **Hal yang Perlu Diperhatikan:** Pastikan menggunakan folder `Variant-b (MultiLabel Classification)`, bukan folder multiclass atau object detection. Setiap gambar harus dicocokkan dengan seluruh labelnya melalui file anotasi. Karena satu gambar bisa memiliki beberapa label, proses evaluasi perlu menggunakan metrik multilabel seperti Micro-F1, Macro-F1, Hamming Loss, dan Subset Accuracy.

### 3.2 Multi-label Bird Species Classification – NIPS 2013

- **Sumber Dataset:** [Kaggle – Multi-label Bird Species Classification NIPS 2013](https://www.kaggle.com/competitions/multilabel-bird-species-classification-nips2013)
- **Penyelenggara:** NIPS4B (Neural Information Processing Scaled for Bioacoustics).
- **Jumlah Data:** Terdiri dari 1.000 rekaman suara alam yang direkam dari berbagai lokasi di Provence, Prancis.
- **Objek Pengamatan:** Rekaman audio lingkungan alam yang mengandung suara burung dan organisme lain.
- **Variabel Target:** Daftar kelas suara yang menunjukkan keberadaan spesies burung atau kelompok suara tertentu dalam rekaman.
- **Jenis Permasalahan:** Multilabel Classification karena satu rekaman audio dapat mengandung suara dari beberapa kelas organisme sekaligus.
- **Jumlah Label:** 87 kelas suara, mencakup burung dan amfibi serta kelas suara ekosistem lainnya.
- **Karakteristik Dataset:** Data utama berupa rekaman audio kontinu dengan durasi yang bervariasi. Model perlu mengenali pola suara untuk menentukan kelas organisme yang muncul dalam setiap rekaman.
- **Hal yang Perlu Diperhatikan:** Dataset ini merupakan data audio, bukan gambar maupun video. Pengolahan dapat dilakukan dengan mengekstraksi fitur audio seperti Mel-Frequency Cepstral Coefficients (MFCC), Mel Spectrogram, dan fitur frekuensi lainnya. Evaluasi dapat menggunakan AUC, Micro-F1, Macro-F1, serta metrik multilabel lainnya. Kompetisi Kaggle ini merupakan kompetisi lama sehingga akses file perlu diperiksa melalui akun Kaggle.

### 3.3 [JOAI Mock Task1] Movie Genre Multimodal

- **Sumber Dataset:** [Kaggle – JOAI Mock Task1 Movie Genre Multimodal](https://www.kaggle.com/competitions/joai-movie-genre-multimodal)
- **Dataset Asli:** MM-IMDb (Multimodal Movie Dataset).
- **Jumlah Data:** 15.552 film pada data training dan 7.799 film pada data testing.
- **Objek Pengamatan:** Poster film dan ringkasan cerita (plot summary) berbahasa Inggris.
- **Variabel Target:** `genres`, yaitu daftar genre yang menggambarkan kategori suatu film.
- **Jenis Permasalahan:** Multilabel Classification karena satu film dapat memiliki beberapa genre secara bersamaan, misalnya Action dan Comedy atau Drama dan Romance.
- **Jumlah Label:** 23 genre, meliputi Drama, Comedy, Romance, Thriller, Crime, Action, Adventure, Horror, Documentary, Mystery, Sci-Fi, Fantasy, Family, Biography, War, History, Music, Animation, Musical, Western, Sport, Short, dan Film-Noir.
- **Representasi Target:** Selain kolom `genres` yang berisi nama genre dipisahkan dengan tanda titik koma, dataset menyediakan 23 kolom binary. Setiap kolom menunjukkan apakah film termasuk dalam genre tertentu (1) atau tidak (0).
- **Karakteristik Dataset:** Menggabungkan dua jenis informasi, yaitu visual dari poster film dan informasi tekstual dari ringkasan cerita. Dengan demikian, dataset ini termasuk multimodal karena memanfaatkan lebih dari satu sumber informasi untuk melakukan prediksi.
- **Hal yang Perlu Diperhatikan:** Kolom `movie_id` digunakan sebagai identitas film, `plot` sebagai informasi teks, dan kolom genre sebagai target. Poster tersedia dalam format JPG. Data training memiliki label, sedangkan data testing tidak menyediakan label target. Pengolahan dapat dilakukan melalui ekstraksi fitur gambar dan representasi teks menggunakan TF-IDF atau model bahasa. Evaluasi menggunakan Macro-F1 sesuai ketentuan kompetisi.

---

## 4. RINGKASAN IDENTIFIKASI DATASET

| Kategori | Dataset | Media Data | Jumlah Kelas/Label |
|---|---|---|---|
| Binary | Breast Cancer Wisconsin | Tabular | 2 |
| Binary | Heart Disease UCI | Tabular | 2 |
| Binary | Credit Card Fraud Detection | Tabular | 2 |
| Multiclass | Dry Bean Classification | Tabular | 7 |
| Multiclass | Rice Seed Pakistan | Image | 7 |
| Multiclass | CI Letter Recognition | Tabular | 26 |
| Multilabel | Tomato-Village | Image | 8 |
| Multilabel | Bird Species NIPS 2013 | Audio | 87 |
| Multilabel | Movie Genre Multimodal | Image dan Teks | 23 |

## 5. KESIMPULAN

Berdasarkan karakteristik target dan struktur data, sembilan dataset yang dipilih termasuk ke dalam permasalahan supervised learning dengan pendekatan klasifikasi.

Ketiga dataset Binary Classification memiliki dua kemungkinan kategori pada target, sedangkan dataset Multiclass Classification memiliki lebih dari dua kategori dengan satu label untuk setiap observasi. Adapun dataset Multilabel Classification memungkinkan satu data mempunyai beberapa label sekaligus.

Tidak terdapat dataset regresi dalam daftar pilihan karena seluruh variabel target yang digunakan berbentuk kategori atau label. Perbedaan utama dari kesembilan dataset terletak pada jenis media, jumlah kelas, struktur fitur, dan metode pengolahan yang diperlukan sebelum proses pemodelan.

## 6. SUMBER REFERENSI

1. [Breast Cancer Wisconsin – Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
2. [Heart Disease UCI – Kaggle](https://www.kaggle.com/competitions/heart-disease-uci)
3. [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
4. [Dry Bean Classification – Kaggle](https://www.kaggle.com/code/deanbatur/dry-bean-classification)
5. [Rice Seed Image Dataset – Kaggle](https://www.kaggle.com/datasets/ishaqedu/rice-seed-image-dataset-from-pakistan)
6. [CI Letter Recognition – Kaggle](https://www.kaggle.com/competitions/ci-letter-recognition)
7. [Tomato-Village – Kaggle](https://www.kaggle.com/datasets/mamtag/tomato-village)
8. [Multi-label Bird Species Classification – Kaggle](https://www.kaggle.com/competitions/multilabel-bird-species-classification-nips2013)
9. [JOAI Movie Genre Multimodal – Kaggle](https://www.kaggle.com/competitions/joai-movie-genre-multimodal)
