# Telecom Customer Churn Prediction

## Deskripsi Proyek
Proyek ini merupakan tugas praktikum mata kuliah Machine Learning yang bertujuan untuk menganalisis dan memprediksi customer churn pada perusahaan telekomunikasi menggunakan Python dan Jupyter Notebook.

Customer churn adalah kondisi ketika pelanggan berhenti menggunakan layanan perusahaan. Melalui proyek ini, dilakukan pengolahan data dan pembangunan model machine learning untuk mempelajari pola pelanggan yang berhenti berlangganan.

## Sumber Dataset
Dataset diperoleh dari Kaggle:

https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics

Dataset awal terdiri dari 7.043 baris dan 38 kolom yang memuat informasi pelanggan, seperti usia, lama berlangganan, biaya bulanan, layanan yang digunakan, dan status pelanggan.

## Tools dan Library
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Tahapan Analisis
1. Data Collecting
2. Data Loading
3. Data Cleaning
4. Data Transformation
5. Exploratory Data Analysis (EDA)
6. Data Processing dan Data Splitting
7. Pembangunan Model Machine Learning
8. Evaluasi Model

## Algoritma
Model klasifikasi yang diuji:
- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

## Hasil Evaluasi Awal
Berdasarkan hasil pengujian lima model klasifikasi:
- Random Forest: accuracy 81,74%
- Logistic Regression: accuracy 78,37%
- Decision Tree: accuracy 73,17%
- KNN: accuracy 67,70%
- SVM: accuracy 52,81%

Random Forest memperoleh accuracy tertinggi di antara lima model tersebut. Namun, Logistic Regression memiliki recall kelas churn sebesar 0,82, sehingga pemilihan model terbaik juga perlu mempertimbangkan tujuan deteksi pelanggan churn.

## Struktur Repository
- `Telecom_Customer_Churn.ipynb`: kode program dan hasil analisis.
- `telecom_customer_churn.csv`: dataset utama.
- `random_forest_model.joblib`: file model Random Forest yang disimpan.
- `README.md`: dokumentasi proyek.

## Cara Menjalankan
1. Unduh atau clone repository ini.
2. Pastikan Python dan Jupyter Notebook telah terpasang.
3. Instal library yang diperlukan.
4. Pastikan dataset tersedia pada lokasi yang sesuai dengan kode.
5. Buka file `Telecom_Customer_Churn.ipynb`.
6. Jalankan sel kode secara berurutan.

## Tujuan Pembelajaran
Proyek ini membantu memahami proses pengolahan data, eksplorasi data, pembangunan model klasifikasi, evaluasi model, serta penyimpanan model machine learning.
