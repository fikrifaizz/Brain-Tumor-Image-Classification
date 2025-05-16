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

Dalam dataset tersebut berisi dua (2) berkas `CSV` yaitu `movies.csv` dan `ratings.csv`. Selanjutnya akan dilakukan *Exploratory Data Analysis*.

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
	- Dataset Movies

 		<img width="442" alt="image" src="https://github.com/user-attachments/assets/9ddf166e-0998-438a-a365-916559234dcc" />

   	- Dataset Ratings

	  	<img width="622" alt="image" src="https://github.com/user-attachments/assets/79906c37-e7b6-4536-972d-fdd480cfe680" />

4. Pengecekan Missing Value

   	- Dataset Movies

   		<img width="113" alt="image" src="https://github.com/user-attachments/assets/ad075bc9-b26d-426e-bd4c-940458099c48" />

	- Dataset Ratings

		<img width="144" alt="image" src="https://github.com/user-attachments/assets/76dd38cf-96f4-45e5-8c0c-b2f35cc08bed" />

	Pada kedua dataset, tidak ditemukan nilai null.

5. Analisis Fitur Genre

<img src="https://github.com/user-attachments/assets/334218b1-4e97-4482-b0ff-2b45551f289b" alt="Analisis Fitur Genre" title="Analisis Fitur Genre">

Berdasarkan visualisasi grafik, Genre Drama merupakan Distribusi terbanyak berdasarkan Jumlah film.

6. Analisis Distribusi Rating

<img src="https://github.com/user-attachments/assets/c1333a7d-e175-4366-b0cc-b3acb24a4fb9" alt="Analisis Distribusi Rating" title="Analisis Distribusi Rating">

Berdasarkan visualisasi grafik, banyak users yang merating film dengan nilai 4.0

## ***Data Preprocessing***
Tahap pra-pemrosesan data atau data preprocessing merupakan tahap yang perlu diterapkan sebelum melakukan proses pemodelan. Tahap ini adalah teknik yang digunakan untuk mengubah data mentah (raw data) menjadi data yang bersih (clean data) yang siap untuk digunakan pada proses selanjutnya. Dalam kasus ini, tahap data preprocessing pada data `movies` dilakukan dengan mengekstrak tahun dari judul film, encoding genre menggunakan one-hot encoding, lalu tahap data preprocessing pada data `ratings` dilakukan dengan konversi timestamp ke `datetime`.
1. Mengekstrak Tahun dari Judul Film pada Dataset `movies`

	Melakukan ekstrak tahun dari judul film pada fitur `title` dan membuat kolom baru yaitu `year`.

 	<img width="502" alt="image" src="https://github.com/user-attachments/assets/46f02f76-5554-49f1-ac83-0f72f2151a7c" />

2. One-hot encoding Genre pada data `movies`

  	Memisahkan genre yang awalnya digabungkan dengan string `|` lalu dipisah dan dijadikan one-hot encoding. One-hot encoding ini sangat penting karena mengubah data kategorikal genre menjadi format numerik yang dapat diproses oleh algoritma machine learning untuk menentukan kemiripan antar film berdasarkan genrenya.

   	<img width="729" alt="image" src="https://github.com/user-attachments/assets/5d3a506f-a077-476a-ac0f-39428e6b2dc3" />

3. Konversi Timestamp ke Datetime pada data `ratings`

   <img width="388" alt="image" src="https://github.com/user-attachments/assets/2ffe4b2b-8c8e-4fa3-bc69-19e4753b6d77" />

   Fitur timestamp sudah dikonversikan ke tipe data `datetime`.

## ***Data Preparation***
Tahap persiapan data atau data preparation juga merupakan tahapan penting sebelum memasuki proses pengembangan model machine learning. Dalam kasus ini, tahap data preparation dilakukan dengan mengatasi missing value, mengatasi data duplikat pada data yang baru di praproses sebelumnya.
1. Mengatasi *Missing Value*

   Karena sebelumnya pada dataset movie dilakukan ekstraksi tahun dari judul film, maka akan dilakukan kembali pengecekan *missing value* karena pada judul film ada judul yang menyertakan tahun dan tidak sehingga ditakutkan adanya nilai null didalamnya.

   Setelah dilakukan pengecekan *missing value*, terdapat nilai null pada fitur `year` sebanyak 13 sehingga akan dilakukan pengisian nilai null tersebut menjadi angka 0.

2. Mengatasi Data Duplikat

   Pada tahapan ini akan dilakukan pengecekan nilai duplikat untuk kedua data yaitu `movies` dan `ratings`, dari pengecekan terhadap dua data tersebut tidak ada nilai duplikat.

3. Menggabungkan semua fitur `genre` pada Data `movies`

   sebelumnya dilakukan pemisahan menggunakan one hot encoding, sehingga fitur yang terdapat nilai 1 akan digabungkan dan disatukan ke kolom baru yaitu `genres_combined`.

   <img width="401" alt="image" src="https://github.com/user-attachments/assets/f4e05271-f1a3-47af-b466-3012a29c68f6" />


## Modelling 
Tahap selanjutnya adalah proses modeling atau membuat model machine learning yang dapat digunakan sebagai sistem rekomendasi untuk menentukan rekomendasi film yang terbaik kepada pengguna dengan beberapa algoritma sistem rekomendasi tertentu.
1. *Content-Based Filtering Recommendation*

   Sistem rekomendasi yang berbasis konten (Content-based Recommendation) adalah sistem rekomendasi yang merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Content-based filtering akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna.

   - *TF-IDF Vectorizer*

     TF-IDF mengubah daftar genre film menjadi nilai numerik dengan mempertimbangkan keunikan setiap genre. Genre umum seperti Drama diberi bobot rendah, sedangkan genre langka seperti Western diberi bobot tinggi. Proses ini menciptakan "sidik jari numerik" untuk setiap film berdasarkan komposisi genrenya. Saat mencari rekomendasi, sistem membandingkan sidik jari ini menggunakan cosine similarity untuk menemukan film dengan karakteristik genre serupa, menghasilkan rekomendasi yang lebih relevan daripada pencocokan genre sederhana.

     <img width="770" alt="image" src="https://github.com/user-attachments/assets/62ed276d-45e2-4123-bb15-8eb767c14a24" />

     Nilai dalam tabel menunjukkan bobot TF-IDF yang mengindikasikan seberapa penting suatu genre untuk film tertentu. Semakin tinggi nilainya, semakin relevan genre tersebut untuk film itu.

   - *Cosine Similarity*

     Cosine similarity mengukur kedekatan antar film berdasarkan vektor TF-IDF genre mereka. Sistem menghitung sudut antara setiap pasangan vektor film dalam ruang multi-dimensi genre. Nilai mendekati 1 menunjukkan film sangat mirip dalam komposisi genre, sedangkan nilai mendekati 0 menandakan film sangat berbeda. Hasil perhitungan tersimpan dalam matriks persegi, di mana setiap sel mewakili tingkat kemiripan antara dua film. Matriks ini menjadi dasar sistem rekomendasi untuk menampilkan film-film yang paling mirip dengan film yang disukai pengguna, menciptakan rekomendasi yang relevan dan personal.

     <img width="715" alt="image" src="https://github.com/user-attachments/assets/a7300aa2-8477-4de9-b5a3-a37eeae7a176" />

   - Hasil *Top N Recommendation*

     Hasil pengujian sistem rekomendasi dengan pendekatan content-based recommendation adalah sebagai berikut.

     <img width="573" alt="image" src="https://github.com/user-attachments/assets/915bb7d1-844c-49f7-991f-95a660b9b507" />

     Pada gambar di atas merupakan data berdasarkan judul film yang dipilih oleh pengguna.

     <img width="758" alt="image" src="https://github.com/user-attachments/assets/78cf4406-965f-49eb-af2f-53573c97f9af" />

     Dapat dilihat bahwa sistem yang telah dibangun berhasil memberikan rekomendasi beberapa judul film berdasarkan input atau masukan sebuah judul film, yaitu "Antz", dan diperoleh beberapa judul film yang berdasarkan perhitungan sistem.

2. *Collaborative Filtering Recommendation*

   Sistem rekomendasi penyaringan kolaboratif (Collaborative Filtering Recommendation) adalah sistem rekomendasi yang merekomendasikan item yang mirip dengan preferensi pengguna di masa lalu, misalnya berdasarkan rating yang telah diberikan oleh pengguna di masa lalu.

   - *Data Preparation*

     Pada tahap ini, perlu melakukan persiapan data untuk menyandikan (encode) fitur `userId` dan `movieId` ke dalam indeks integer lalu memetakan `userId` dan `movieId` ke dalam masing-masing dataframe yang berkaitan.

     Diperoleh jumlah user sebesar 610, jumlah film sebesar 9724, nilai minimal rating yaitu 0.5, dan nilai maksimum rating yaitu 5.
   - *Split Training and Validation Data*

     Melakukan pembagian dataset dengan rasio 80:20, yaitu 80% untuk data latih (*training data*) dan 20% untuk data uji (*validation data*).

   - *Model Development* dan Hasil

     Berdasarkan model yang telah di-training, berikut adalah hasil pengujian sistem rekomendasi film dengan pendekatan collaborative filtering recommendation.

     <img width="757" alt="image" src="https://github.com/user-attachments/assets/1aa5e63c-6031-4f7e-8520-9e5a0008fb75" />

     Berdasarkan hasil di atas, dapat dilihat bahwa sistem akan mengambil pengguna secara acak, yaitu pengguna dengan user_id 1. Lalu akan dicari film dengan rating terbaik dari user tersebut. Kemudian sistem akan membandingan antara film dengan rating tertinggi dari user dan semua film, kecuali film yang telah dibaca tersebut, lalu akan mengurutkan film yang akan direkomendasikan berdasarkan nilai rekomendasi yang tertinggi. Dapat dilihat terdapat 5 daftar film yang direkomendasikan oleh sistem.
     - Secrets & Lies (Prediksi Rating: 4.89)
     - Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (Prediksi Rating: 4.87)
     - High Noon (Prediksi Rating: 4.87)
     - Celebration, The (Festen) (Prediksi Rating: 4.86)
     - Touch of Evil (Prediksi Rating: 4.86)

## Evaluation
1. *Content-Based Filtering Recommendation

   Pada tahap evaluasi untuk model sistem rekomendasi dengan pendekatan content-based recommendation dapat menggunakan evaluasi dengan metrik akurasi yang diperoleh dari,

   $Accuracy = \frac{\sum_{i=1}^{n} RecommendedMovie_{i}}{\sum_{i=1}^{n}MovieWithSameGenres_{i}} \times 100$

   Masih menggunakan data yang sama pada tahap Modeling content-based recommendation, pada proses Hasil Top-N Recommendation, yaitu judul film "Antz", akan dilakukan proses pencarian jumlah judul film dengan `movieId` yang sama. Pencarian tersebut menggunakan variabel baru yang di mana akan mengambil sebuah data film dengan genres yang sama. Hasil yang diperoleh adalah Antz memiliki jumlah film sebanyak 13 genres yang mirip.

   Proses perhitungan akurasi dilakukan dengan membagi banyaknya rekomendasi film yang dihasilkan, dibagi dengan banyaknya jumlah film dengan genres yang sama, kemudian dikalikan dengan 100. Sehingga diperoleh nilai akurasi sebesar 38.46%.

2. *Collaborative Filtering Recommendation*

   Berdasarkan model machine learning yang sudah dibangun menggunakan embedding layer dengan Adam optimizer dan binary crossentropy loss function, metrik yang digunakan adalah Root Mean Squared Error (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut,

   $RMSE = \sqrt{\frac{\sum_{i=1}^{n} (y_i - y\_pred_i)^2}{n}}$

   Di mana, nilai $n$ merupakan jumlah dataset, nilai $y_i$ adalah nilai sebenarnya, dan y_pred yaitu nilai prediksinya terdahap $i$ sebagai urutan data dalam dataset.

   Hasil nilai RMSE yang rendah menunjukkan bahwa variasi nilai yang dihasilkan dari model sistem rekomendasi mendekati variasi nilai observasinya. Artinya, semakin kecil nilai RMSE, maka akan semakin dekat nilai yang diprediksi dan diamati.

   Berikut merupakan visualisasi hasil training dan validation error dari metrik RMSE serta training dan validation loss ke dalam grafik plot.

   <img src="https://github.com/user-attachments/assets/03efb52d-8d4a-43e7-bb5f-50c7489ec5cd" alt="training dan validation loss" title="Dtraining dan validation loss">

## Kesimpulan
Kesimpulannya adalah model yang digunakan untuk melakukan rekomendasi buku berdasarkan teknik Content-based Recommendation dan teknik Collaborative Filtering Recommendation telah berhasil dibuat dan sesuai dengan preferensi pengguna. Pada collaborative filtering diperlukan data rating dari pengguna, sedangkan pada content-based filtering, data rating tidak diperlukan karena analisis sistem rekomendasi akan berdasarkan atribut genres dari masing-masing film.



   





     


     
	
	 	





Referensi :

[1] Resnick, P., et al. (1994). GroupLens: An Open Architecture for Collaborative Filtering of Netnews.

[2] Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context.

[3] Netflix Tech Blog (2017). Recommending for the World.

[4] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0

[5] Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.

[6] A. Ajitsaria, "Build a Recommendation Engine With Collaborative Filtering", Real Python, 2019, Retrieved from: https://realpython.com/build-recommendation-engine-collaborative-filtering.
