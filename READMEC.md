# Laporan Proyek Machine Learning - Ica Nur Halimah
## Project Overview
Proyek ini membahas permasalahan membanjirnya produk fashion di era digital yang menyulitkan pengguna dalam menentukan pilihan, serta bagaimana penerapan sistem rekomendasi berbasis data dapat memberikan solusi yang relevan dan efisien.

<img src="https://github.com/user-attachments/assets/840410c4-d3bc-4cc0-82bd-fff9adf1b27f" alt="Ilustrasi Netflix" title="Ilustrasi Netflix">

Di era digital, pertumbuhan industri e-commerce berkembang pesat, khususnya pada sektor fashion. Ribuan produk dengan variasi merek, kategori, warna, ukuran, hingga harga ditawarkan setiap harinya melalui berbagai platform. Banyaknya pilihan ini justru memunculkan tantangan baru bagi pengguna: kesulitan dalam menemukan produk yang sesuai dengan selera dan kebutuhannya secara efisien. Fenomena ini dikenal sebagai information overload, yaitu kondisi ketika konsumen kebingungan akibat terlalu banyak informasi dan pilihan [1]. Oleh karena itu, dibutuhkan sistem yang mampu memberikan rekomendasi produk secara personal dan relevan untuk membantu pengguna dalam mengambil keputusan secara cepat dan tepat.

Sistem rekomendasi telah menjadi elemen penting dalam meningkatkan pengalaman pengguna (user experience) serta retensi pelanggan di berbagai platform digital. Dalam konteks e-commerce, sistem ini tidak hanya mempermudah pengguna, tetapi juga berdampak langsung terhadap peningkatan konversi penjualan [2]. Dengan menerapkan pendekatan Content-Based Filtering dan Collaborative Filtering, proyek ini bertujuan membangun model rekomendasi yang mampu menyarankan produk fashion berdasarkan preferensi individual dan pola perilaku pengguna lain.

Menurut laporan McKinsey, lebih dari 35% pembelian di Amazon berasal dari sistem rekomendasi yang mereka terapkan [3]. Selain itu, penelitian oleh Jannach et al. (2016) menunjukkan bahwa pendekatan gabungan (hybrid) antara konten dan kolaboratif dapat meningkatkan akurasi rekomendasi secara signifikan dalam domain fashion [4]. Hal ini memperkuat relevansi pendekatan yang digunakan dalam proyek ini, yakni mengintegrasikan dua teknik utama untuk mencapai hasil rekomendasi yang optimal.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:

1. Bagaimana cara melakukan tahap persiapan data fashion agar dapat digunakan sebagai informasi untuk membuat model machine learning sistem rekomendasi?
2. Bagaimana cara membuat model machine learning untuk sistem rekomendasi fashion?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:

1. Melakukan tahap persiapan data sehingga data siap digunakan pada model machine learning untuk sistem rekomendasi.
2. Membuat model machine learning untuk sistem rekomendasi fashion terbaik kepada pengguna.

### Solution Statements
Berdasarkan tujuan dari proyek yang telah dipaparkan di atas, maka berikut adalah beberapa solusi yang dapat dilakukan agar dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap pra-pemrosesan data atau data preprocessing merupakan tahap untuk mengubah data mentah atau raw data menjadi data yang bersih atau clean data yang siap untuk digunakan pada proses selanjutnya. Tahap ini dapat dilakukan dengan cara, yaitu:
   - Konversi Fitur Rating.
   - Menambah Fitur Kategori untuk `Price`.
2. Tahap persiapan data atau data preparation merupakan proses transformasi pada data sehingga data menjadi bentuk yang cocok untuk melakukan proses pemodelan di tahap selanjutnya. Tahap ini dapat dilakukan dengan beberapa teknik, yaitu:
   - Melakukan pengecekan nilai data yang kosong, tidak ada, ataupun null (missing value) dan menghapus data tersebut atau mengganti/mengisinya dengan suatu nilai tertentu.
   - Melakukan pengecekan data yang mungkin duplikat agar tidak akan mengganggu hasil dari pemodelan dan sistem yang telah dibangun.
   - Membuat fitur tekstual untuk TF-IDF.
3.  Tahap pembuatan model machine learning untuk sistem rekomendasi film adalah jenis algoritma sistem rekomendasi yang terpersonalisasi atau personalized recommender system. Pembuatan model akan menggunakan dua (2) pendekatan, yaitu content-based filtering recommendation, dan pendekatan collaborative filtering recommendation.
   - ***Content-based filtering recommendation***

     Sistem rekomendasi yang berbasis konten (content-based filtering) merupakan sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan item yang disukai oleh pengguna di masa lalu. Content-based filtering akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna lain sebelumnya. Pada pendekatan menggunakan content-based filtering akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity.
     - TF-IDF Vectorizer

       TF-IDF adalah metode untuk mengubah teks menjadi vektor numerik berdasarkan seberapa penting sebuah kata terhadap sebuah dokumen dalam koleksi dokumen. Ini digunakan dalam content-based filtering untuk merepresentasikan fitur dari item, dalam kasus ini genre film atau deskripsi lainnya [5].

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

    Collaborative Filtering (CF) merupakan pendekatan dalam sistem rekomendasi yang bekerja dengan menganalisis pola interaksi antar pengguna dan item tanpa bergantung pada informasi konten produk. CF berasumsi bahwa jika dua pengguna memiliki kesamaan preferensi terhadap beberapa item, maka mereka juga akan menyukai item yang belum dilihat namun disukai oleh pengguna lain yang mirip.

    Seiring berkembangnya teknologi, pendekatan ini telah mengalami peningkatan signifikan melalui integrasi dengan Deep Learning, khususnya menggunakan jaringan saraf tiruan. Pendekatan ini disebut Neural Collaborative Filtering (NCF), yang memungkinkan sistem belajar representasi vektor (embedding) dari pengguna dan produk untuk memodelkan hubungan kompleks secara non-linear. Model matematis yang digunakan sebagai berikut [6]:

    $\hat{r}_{uv} = f([u; v])$

    dengan:
    - $u \in \mathbb{R}^k$ adalah embedding vektor pengguna,
    - $v \in \mathbb{R}^k$ adalah embedding vektor produk,
    - $f$ adalah fungsi jaringan saraf
    - $[u; v]$ adalah concatenation dari dua vektor embedding

    Langkah-langkah:
    - Membangun user-item interaction matrix berdasarkan User ID, Product ID, dan Rating.
    - Menggunakan embedding layer untuk merepresentasikan pengguna dan produk ke dalam vektor berdimensi rendah.
    - Kombinasi vektor tersebut diumpankan ke jaringan saraf (multi-layer perceptron) untuk mempelajari hubungan non-linear dan memprediksi rating.


## Data Understanding

<img width="1196" alt="image" src="https://github.com/user-attachments/assets/81325b3f-2274-4150-b26a-ce80596433fe" />

Data yang digunakan dalam proyek ini adalah dataset yang diambil dari Kaggle Dataset. Di bawah ini adalah informasi detail tentang dataset yang digunakan.

|                         | Keterangan                                                                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset: Fashion Products](https://www.kaggle.com/datasets/bhanupratapbiswas/fashion-products) |
| *Usability*             | 10.00                           |
| Lisensi                 | Other |
| Jenis dan Ukuran Berkas | zip (64.09 KB)                  |
| Kategori                | Clothing and Accessories     |

Dalam dataset tersebut berisi satu (1) berkas `CSV` yaitu `fashion_products.csv`. Selanjutnya akan dilakukan *Exploratory Data Analysis*.

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

[1] Bawden, D., & Robinson, L. (2009). The dark side of information: Overload, anxiety and other paradoxes and pathologies. Journal of Information Science, 35(2), 180–191.

[2] Gomez-Uribe, C. A., & Hunt, N. (2016). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 1–19.

[3] McKinsey & Company. (2013). Big data, the next frontier for innovation, competition, and productivity.

[4] Jannach, D., Adomavicius, G., Tuzhilin, A., et al. (2016). Recommender Systems—Challenges, Insights and Research Opportunities. ACM Transactions on Intelligent Systems and Technology, 7(1), 1–34.

[5]

[6] Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep Learning based Recommender System: A Survey and New Perspectives. ACM Computing Surveys (CSUR), 52(1), 1–38.


