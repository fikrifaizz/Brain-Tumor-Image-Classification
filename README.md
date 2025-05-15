# Laporan Proyek Machine Learning - Fikri Faiz Zulfadhli
## Project Overview
Proyek ini membahas permasalahan membanjirnya konten film di era digital yang menyulitkan pengguna dalam menentukan pilihan, serta bagaimana penerapan sistem rekomendasi berbasis data dapat memberikan solusi yang relevan dan efisien.

<img src="https://github.com/user-attachments/assets/db2c587d-7779-4cf3-a731-4939de4dedca" alt="Ilustrasi Netflix" title="Ilustrasi Netflix">

Di era digital saat ini, pengguna internet dihadapkan pada lautan informasi dan pilihan konten yang nyaris tak terbatas. Hal ini terutama terlihat dalam industri hiburan seperti film, di mana platform streaming menawarkan ribuan judul film dari berbagai genre dan negara. Kondisi ini menciptakan kebutuhan akan sistem yang mampu menyaring dan merekomendasikan konten yang relevan sesuai dengan preferensi tiap pengguna. Sistem rekomendasi hadir sebagai solusi, memberikan pengalaman yang dipersonalisasi dan efisien dalam membantu pengguna menemukan film yang sesuai dengan selera mereka [1].

Proyek ini dibangun berdasarkan data dari MovieLens, salah satu dataset benchmark paling populer yang digunakan dalam riset sistem rekomendasi. Dengan menerapkan pendekatan seperti collaborative filtering dan content-based filtering, sistem ini diharapkan mampu mereplikasi kemampuan layanan besar seperti Netflix atau Amazon Prime dalam memberikan rekomendasi film yang akurat. Tidak hanya berperan sebagai alat bantu pengguna, sistem ini juga memberikan nilai strategis bagi pengembang dan pemilik platform dalam meningkatkan engagement dan loyalitas pengguna [2][3].

## Business Understanding
### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:

1. Bagaimana cara melakukan tahap persiapan data movies dan rating agar dapat digunakan sebagai informasi untuk membuat model machine learning sistem rekomendasi?
2. Bagaimana cara membuat model machine learning untuk sistem rekomendasi film?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:

1. Melakukan tahap persiapan data sehingga data siap digunakan pada model machine learning untuk sistem rekomendasi.
2. Membuat model machine learning untuk sistem rekomendasi film terbaik kepada pengguna.

### Solution Statements
Berdasarkan tujuan dari proyek yang telah dipaparkan di atas, maka berikut adalah beberapa solusi yang dapat dilakukan agar dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap pra-pemrosesan data atau data preprocessing merupakan tahap untuk mengubah data mentah atau raw data menjadi data yang bersih atau clean data yang siap untuk digunakan pada proses selanjutnya. Tahap ini dapat dilakukan dengan cara, yaitu:
   - Mengambil informasi tahun dari judul film (misalnya, Toy Story (1995) menjadi 1995) dan menyimpannya ke dalam kolom tersendiri (year), kemudian menghapus tahun dari kolom title untuk keperluan pemodelan teks yang lebih bersih.
   - Mengubah format data suatu fitur menjadi one-hot encoding
   - Mengkonversi timestamp dari format `UNIX` ke format `datetime`.
2. Tahap persiapan data atau data preparation merupakan proses transformasi pada data sehingga data menjadi bentuk yang cocok untuk melakukan proses pemodelan di tahap selanjutnya. Tahap ini dapat dilakukan dengan beberapa teknik, yaitu:
   - Melakukan pengecekan nilai data yang kosong, tidak ada, ataupun null (missing value) dan menghapus data tersebut atau mengganti/mengisinya dengan suatu nilai tertentu.
   - Melakukan pengecekan data yang mungkin duplikat agar tidak akan mengganggu hasil dari pemodelan dan sistem yang telah dibangun.
   - Menggabungkan data.
3.  Tahap pembuatan model machine learning untuk sistem rekomendasi film adalah jenis algoritma sistem rekomendasi yang terpersonalisasi atau personalized recommender system. Pembuatan model akan menggunakan dua (2) pendekatan, yaitu content-based filtering recommendation, dan pendekatan collaborative filtering recommendation.
   - ***Content-based filtering recommendation***

     Sistem rekomendasi yang berbasis konten (content-based filtering) merupakan sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan item yang disukai oleh pengguna di masa lalu. Content-based filtering akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna lain sebelumnya. Pada pendekatan menggunakan content-based filtering akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity.
     - TF-IDF Vectorizer

       TF-IDF adalah metode untuk mengubah teks menjadi vektor numerik berdasarkan seberapa penting sebuah kata terhadap sebuah dokumen dalam koleksi dokumen. Ini digunakan dalam content-based filtering untuk merepresentasikan fitur dari item, dalam kasus ini genre film atau deskripsi lainnya [4].

       Untuk kata $t$ dalam dokumen $d$ dari kumpulan dokumen $D$:

       $TF-IDF(t,d)=TF(t,d) \times IDF(t)$

       dengan:
       - TF (Term Frequency):

         $TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$

         Di mana $f_{t,d}$ adalah jumlah kemunculan kata $t$ dalam dokumen $d$.
       - IDF (Inverse Document Frequency):

         $\text{IDF}(t) = \log\left(\frac{N}{1 + |\{d \in D : t \in d\}|}\right)$

         Di mana:
         
	        - $N$ adalah jumlah total dokumen.
          - $|\{d \in D : t \in d\}|$ adalah jumlah dokumen yang mengandung kata $t$.

        Cara Kerja:
       1. Pisahkan teks genres menjadi kata-kata individual.
       2. Hitung frekuensi tiap kata (TF) dalam setiap item.
       3. Hitung IDF untuk setiap kata di seluruh koleksi film.
       4. Kalikan TF dan IDF untuk mendapatkan bobot penting dari setiap kata di tiap film.

      - Cosine Similarity

        Setelah setiap item (film) direpresentasikan sebagai vektor TF-IDF, kita dapat mengukur kemiripan antar film menggunakan cosine similarity. Rumus matematisnya sebagai berikut [5]:

        $\text{Cosine Similarity} (A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$

        Dengan:
        - $A \cdot B$ adalah hasil dot product antara vektor $A$ dan $B$.
        - $\|A\|$ adalah panjang vektor $A$ (norma).
        - Nilai hasil berada di antara 0 (tidak mirip) hingga 1 (sangat mirip).
  - ***Collaborative filtering recommendation***

    Sistem rekomendasi yang berbasis penyaringan kolaboratif (collaborative filtering) adalah sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan preferensi pengguna di masa lalu berdasarkan riwayat pengguna lain yang memiliki preferensi yang sama, misalnya berdasarkan penilaian atau rating yang telah diberikan pengguna di masa lalu. [6] Namun, teknik ini memilki kekurangan yaitu, tidak dapat memberikan rekomendasi item yang tidak memiliki riwayat penilaian/rating atau transaksi.

	Menggunakan teknik collaborative filtering recommendation akan memerlukan proses penyandian (encoding) fitur-fitur yang terdapat pada dataset ke dalam bentuk indeks integer, lalu memetakannya ke dalam dataframe yang berkaitan. Kemudian akan dilakukan pembagian distribusi dataset dengan rasio tertentu untuk memisahkan data latih (training data) dan juga data uji (validation data) sebelum dilakukan tahap pemodelan.

## Data Understanding

<img src="https://github.com/user-attachments/assets/34545728-3662-4c23-83c0-6b68108ed882" alt="Dataset Kaggle" title="Dataset Kaggle">

Data yang digunakan dalam proyek ini adalah dataset yang diambil dari Kaggle Dataset. Di bawah ini adalah informasi detail tentang dataset yang digunakan.

|                         | Keterangan                                                                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset: Movies & Ratings for Recommendation System](https://github.com/user-attachments/assets/59682e60-a4e3-4352-be67-626c4ca11ceb) |
| *Usability*             | 9.41                           |
| Lisensi                 | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| Jenis dan Ukuran Berkas | zip (2.98 MB)                  |
| Kategori                | Movies and TV Shows, knn       |

Dalam dataset tersebut berisi dua (2) berkas `CSV` yaitu `movies.csv` dan `ratings.csv`.

1. Mengecek Jumlah Data Masing-masing Atribut dari Dataset

   <img width="486" alt="image" src="https://github.com/user-attachments/assets/32456857-26dd-4bf5-94ab-49ac6be3fd22" />

2. Deskripsi Variabel

   - Dataset Movies

  	 	<img width="299" alt="image" src="https://github.com/user-attachments/assets/1814b5c2-6d30-4bab-9c81-40a0c82a4dc8" />

   		- `movieID` : Nomor identifikasi unik untuk setiap film.
   		- `title` : Judul film beserta tahun rilisnya.
   		- `genres` : Kategori genre film.


   - Dataset Ratings

	 	<img width="330" alt="image" src="https://github.com/user-attachments/assets/c3710c5a-2a9d-44a4-bb27-2dd2b68fd863" />

   		- `userId` : Identifikasi pengguna.
     	- `movieId` : Identifikasi film.
     	- `rating` : Nilai penilaian yang diberikan pengguna
     	- `timestamp` : Waktu ketika rating diberikan
3. Deskripsi Statistik

	 	





Referensi :

[1] Resnick, P., et al. (1994). GroupLens: An Open Architecture for Collaborative Filtering of Netnews.

[2] Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context.

[3] Netflix Tech Blog (2017). Recommending for the World.

[4] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0

[5] Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.

[6] A. Ajitsaria, "Build a Recommendation Engine With Collaborative Filtering", Real Python, 2019, Retrieved from: https://realpython.com/build-recommendation-engine-collaborative-filtering.
