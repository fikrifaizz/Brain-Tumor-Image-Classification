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
3.  Tahap pembuatan model machine learning untuk sistem rekomendasi fashion adalah jenis algoritma sistem rekomendasi yang terpersonalisasi atau personalized recommender system. Pembuatan model akan menggunakan dua (2) pendekatan, yaitu content-based filtering recommendation, dan pendekatan collaborative filtering recommendation.
   - ***Content-based filtering recommendation***

     Sistem rekomendasi yang berbasis konten (content-based filtering) merupakan sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan item yang disukai oleh pengguna di masa lalu. Content-based filtering akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna lain sebelumnya. Pada pendekatan menggunakan content-based filtering akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity.
     - TF-IDF Vectorizer

       TF-IDF adalah metode untuk mengubah teks menjadi vektor numerik berdasarkan seberapa penting sebuah kata terhadap sebuah dokumen dalam koleksi dokumen. Ini digunakan dalam content-based filtering untuk merepresentasikan fitur dari item, dalam kasus ini fashion atau deskripsi lainnya [5].

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
       3. Hitung IDF untuk setiap kata di seluruh produk.
       4. Kalikan TF dan IDF untuk mendapatkan bobot penting dari setiap kata di tiap produk.

      - Cosine Similarity

        Setelah setiap produk direpresentasikan sebagai vektor TF-IDF, kita dapat mengukur kemiripan antar fashion menggunakan cosine similarity. Rumus matematisnya sebagai berikut [6]:

        $\text{Cosine Similarity} (A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$

        Dengan:
        - $A \cdot B$ adalah hasil dot product antara vektor $A$ dan $B$.
        - $\|A\|$ adalah panjang vektor $A$ (norma).
        - Nilai hasil berada di antara 0 (tidak mirip) hingga 1 (sangat mirip).
  - ***Collaborative filtering recommendation***

    Collaborative Filtering (CF) merupakan pendekatan dalam sistem rekomendasi yang bekerja dengan menganalisis pola interaksi antar pengguna dan item tanpa bergantung pada informasi konten produk. CF berasumsi bahwa jika dua pengguna memiliki kesamaan preferensi terhadap beberapa item, maka mereka juga akan menyukai item yang belum dilihat namun disukai oleh pengguna lain yang mirip.

    Seiring berkembangnya teknologi, pendekatan ini telah mengalami peningkatan signifikan melalui integrasi dengan Deep Learning, khususnya menggunakan jaringan saraf tiruan. Pendekatan ini disebut Neural Collaborative Filtering (NCF), yang memungkinkan sistem belajar representasi vektor (embedding) dari pengguna dan produk untuk memodelkan hubungan kompleks secara non-linear. Model matematis yang digunakan sebagai berikut [7]:

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

   <img width="370" alt="image" src="https://github.com/user-attachments/assets/48d16ec2-8bcb-4e23-8c38-e00191d1724a" />

2. Deskripsi Variabel

   <img width="351" alt="image" src="https://github.com/user-attachments/assets/8fe3d9ef-9a12-42a5-8b37-08735924147e" />

   - `User ID` : Nomor identifikasi pengguna yang berinteraksi dengan produk.
   - `Product ID` : Nomor identifikasi unik untuk setiap produk dalam database.
   - `Product Name` : Nama produk fashion.
   - `Brand` : Merek produk fashion.
   - `Category` : Kategori target pasar produk.
   - `Price` : Harga produk dalam suatu satuan mata uang.
   - `Rating` : Penilaian produk, tampaknya dalam skala angka desimal.
   - `Color` : Warna Produk.
   - `Size` : Ukuran produk.

3. Deskripsi Statistik

   <img width="315" alt="image" src="https://github.com/user-attachments/assets/335ad85b-a695-4dfa-ad9e-7b1235e97781" />

4. Pengecekan Missing Value

   <img width="160" alt="image" src="https://github.com/user-attachments/assets/60f81658-e7ec-4573-bff0-82e743e38671" />

   Tidak terdapat nilai null di setiap fitur.

5. Analisis Distribusi Fitur Produk

   <img width="778" alt="image" src="https://github.com/user-attachments/assets/5324e0ad-c85f-492f-a028-296fcf952482" />

   Berdasarkan visualisasi grafik, distribusi kategori produk bisa dikatakan sama rata hanya saja produk `Kids' Fashion` sedikit lebih banyak.

6. Analisis Distribusi Rating

   <img width="784" alt="image" src="https://github.com/user-attachments/assets/6e8c4553-5f3e-482a-899f-a4a294024e44" />

   Berdasarkan Visualisasi Grafik, Frekuensi Rating memiliki jumlah yang hampir sama rata dengan yang tertinggi berada di antara 2.5 - 3.0 dan terendah di antara 2.0 - 2.5.

7. Analisis Distribusi Brand

   <img width="777" alt="image" src="https://github.com/user-attachments/assets/d0cf24d8-8745-4ffd-b34a-3079aeba2489" />

   Berdasarkan visualisasi grafik, distribusi brand bisa dikatakan sama rata hanya saja produk `Nike` sedikit lebih banyak.

8. Analisis Distribusi Harga

   <img width="770" alt="image" src="https://github.com/user-attachments/assets/23149ac9-24b1-4210-859e-643f5c3d5eb5" />

   Berdasarkan visualisasi grafik, harga pada produk memiliki rentang yang beragam namun produk terbanyak berada di nilai 90.

## ***Data Preprocessing***
Tahap pra-pemrosesan data atau data preprocessing merupakan tahap yang perlu diterapkan sebelum melakukan proses pemodelan. Tahap ini adalah teknik yang digunakan untuk mengubah data mentah (raw data) menjadi data yang bersih (clean data) yang siap untuk digunakan pada proses selanjutnya. Dalam kasus ini, tahap data preprocessing pada data `fashion` dilakukan dengan mengonversi fitur rating menjadi `float` dengan 1 angka di belakang koma dan mengubah harga ke katogori.
1. Perubahan Format Fitur `Rating`

   Melakukan perubahan format Fitur Rating agar sesuai dengan umumnya yaitu 1 angka dibelakang koma.

   <img width="181" alt="image" src="https://github.com/user-attachments/assets/2ab75c4d-ae32-4aac-a3c9-b3001382dfd1" />

   Fitur Rating sudah diubah menjadi float dengan 1 angka di belakang koma.

2. Menambah Fitur Kategori untuk `Price`

   Karena nilai harga di dataset berkisar antara 10 hingga 100, maka akan dikelompokkan dengan rentang harga ke dalam tiga kategori: **Murah** (10-30), **Sedang** (31-70), dan **Mahal** (71-100).

   <img width="377" alt="image" src="https://github.com/user-attachments/assets/201a606b-ef2e-49c3-a3fb-7549634a0c27" />

   Kategorisasi ini membantu dalam proses preprocessing untuk memudahkan analisis dan model rekomendasi.

## ***Data Preparation***
Tahap persiapan data atau data preparation juga merupakan tahapan penting sebelum memasuki proses pengembangan model machine learning. Dalam kasus ini, tahap data preparation dilakukan dengan mengatasi missing value, mengatasi data duplikat, menggabungkan fitur yang diperlukan untuk rekomendasi.
1. Mengatasi *Missing Value*

   Pada Data Understanding sudah dilakukan pengecekan *missing value* sehingga tidak didapatkan nilai null untuk setiap fitur yang ada di data.

2. Mengatasi Data Duplikat

   Pada tahapan ini akan dilakukan pengecekan nilai duplikat untuk data, dari pengecekan terhadap dua data tersebut tidak ada nilai duplikat.

3. Membuat fitur tekstual untuk TF-IDF

   Salah satu teknik penting adalah penggabungan fitur-fitur kategorikal menjadi representasi tekstual yang dapat diproses oleh algoritma machine learning.

   <img width="629" alt="image" src="https://github.com/user-attachments/assets/abcc7b4c-5fbe-435a-b5ee-0aa1298cb259" />

   Penggabungan fitur seperti ini memungkinkan sistem untuk memberikan rekomendasi yang lebih personal dan kontekstual, menyesuaikan dengan preferensi pengguna terhadap karakteristik produk yang spesifik seperti kategori harga, merek favorit, kategori produk, warna, dan ukuran yang disukai.

## Modelling 
Tahap selanjutnya adalah proses modeling atau membuat model machine learning yang dapat digunakan sebagai sistem rekomendasi untuk menentukan rekomendasi produk yang terbaik kepada pengguna dengan beberapa algoritma sistem rekomendasi tertentu.
1. *Content-Based Filtering Recommendation*

   Sistem rekomendasi yang berbasis konten (Content-based Recommendation) adalah sistem rekomendasi yang merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Content-based filtering akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna.

   - *TF-IDF Vectorizer*

     *TF-IDF Vectorizer* digunakan untuk mengekstrak fitur penting dari teks dan menghasilkan matriks numerik yang mencerminkan relevansi setiap kata.

     <img width="769" alt="image" src="https://github.com/user-attachments/assets/776cbeb1-7a40-420e-98ed-670e5f711a7b" />

     Hasil dari proses vectorization ini adalah matriks TF-IDF yang menangkap frekuensi dan kepentingan setiap term dalam fitur gabungan. Matriks ini akan menjadi dasar untuk menghitung kemiripan antar produk dalam sistem rekomendasi berbasis konten.

   - *Cosine Similarity*

     Setelah mendapatkan representasi vektor TF-IDF, langkah selanjutnya adalah menghitung derajat kemiripan antar produk menggunakan cosine similarity. Pendekatan ini memungkinkan sistem untuk mengidentifikasi produk-produk dengan karakteristik serupa.

     <img width="669" alt="image" src="https://github.com/user-attachments/assets/275f5da4-dc5e-4697-830f-367d6a213430" />

     Matriks similarity yang dihasilkan menyediakan skor kemiripan antara setiap pasangan produk. Nilai yang lebih tinggi menunjukkan kemiripan yang lebih besar, di mana nilai 1.0 berarti produk identik berdasarkan fitur-fitur yang dianalisis. Matriks ini akan menjadi dasar untuk memberikan rekomendasi produk yang paling sesuai dengan preferensi pengguna.


   - Hasil *Top N Recommendation*

     Untuk memulai proses rekomendasi, kita perlu memilih produk spesifik sebagai referensi. Karena terdapat banyak produk "Jeans" dalam dataset, kita memilih satu produk tertentu berdasarkan indeks untuk mendapatkan rekomendasi yang lebih tepat.
  
     <img width="273" alt="image" src="https://github.com/user-attachments/assets/b874b434-b42a-404f-b9fe-626f54a040bf" />

     Dengan mengidentifikasi produk referensi yang spesifik, sistem rekomendasi dapat memberikan saran yang lebih akurat berdasarkan karakteristik produk tersebut. 

     <img width="770" alt="image" src="https://github.com/user-attachments/assets/4f2c2ea7-ca03-485e-b975-257dd7b1e644" />

     Dapat dilihat bahwa sistem yang telah dibangun berhasil memberikan rekomendasi beberapa produk fashion berdasarkan input atau masukan sebuah nama produk, yaitu "jeans", dan diperoleh beberapa produk fashion yang berdasarkan perhitungan sistem.

3. *Collaborative Filtering Recommendation*

   Sistem rekomendasi penyaringan kolaboratif (Collaborative Filtering Recommendation) adalah sistem rekomendasi yang merekomendasikan item yang mirip dengan preferensi pengguna di masa lalu, misalnya berdasarkan rating yang telah diberikan oleh pengguna di masa lalu.

   - *Data Preparation*

     Pada tahap ini, perlu melakukan persiapan data untuk menyandikan (encode) fitur `User ID` dan `Product ID` ke dalam indeks integer lalu memetakan `User ID` dan `Product ID` ke dalam masing-masing dataframe yang berkaitan.

     Diperoleh jumlah user sebesar 100, jumlah produk sebesar 1000, nilai minimal rating yaitu 1, dan nilai maksimum rating yaitu 5.
   - *Split Training and Validation Data*

     Melakukan pembagian dataset dengan rasio 80:20, yaitu 80% untuk data latih (*training data*) dan 20% untuk data uji (*validation data*).

   - *Model Development* dan Hasil

     Berdasarkan model yang telah di-training, berikut adalah hasil pengujian sistem rekomendasi produk dengan pendekatan collaborative filtering recommendation.

     <img width="621" alt="image" src="https://github.com/user-attachments/assets/76b86527-5836-4ef3-8108-3a99475d40e8" />

     Berdasarkan hasil di atas, dapat dilihat bahwa bahwa model collaborative filtering berhasil mengidentifikasi preferensi pengguna 35 berdasarkan pola rating dari pengguna-pengguna lain yang memiliki selera serupa, dan memberikan rekomendasi produk yang mungkin disukai namun belum dilihat/dibeli oleh pengguna tersebut.

## Evaluation
1. *Content-Based Filtering Recommendation

   Pada tahap evaluasi untuk model sistem rekomendasi dengan pendekatan content-based recommendation dapat menggunakan evaluasi dengan metrik akurasi yang diperoleh dari,

   $Accuracy = \frac{\sum_{i=1}^{n} RecommendedProduct_{i}}{\sum_{i=1}^{n}ProductWithSamePreference_{i}} \times 100$

   Masih menggunakan data yang sama pada tahap Modeling content-based recommendation, pada proses Hasil Top-N Recommendation, yaitu produk dengan nama "jeans", akan dilakukan proses pencarian jumlah produk fashion dengan `Product ID` yang sama. Pencarian tersebut menggunakan variabel baru yang di mana akan mengambil sebuah data fashion dengan preferensi yang sama. Hasil yang diperoleh adalah Jeans memiliki jumlah produk sebanyak 16 produk yang mirip.

   Proses perhitungan akurasi dilakukan dengan membagi banyaknya rekomendasi fashion yang dihasilkan, dibagi dengan banyaknya jumlah fashion dengan preferensi yang sama, kemudian dikalikan dengan 100. Sehingga diperoleh nilai akurasi sebesar 32.00%.

2. *Collaborative Filtering Recommendation*

   Berdasarkan model machine learning yang sudah dibangun menggunakan embedding layer dengan Adam optimizer dan binary crossentropy loss function, metrik yang digunakan adalah Root Mean Squared Error (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut,

   $RMSE = \sqrt{\frac{\sum_{i=1}^{n} (y_i - y\_pred_i)^2}{n}}$

   Di mana, nilai $n$ merupakan jumlah dataset, nilai $y_i$ adalah nilai sebenarnya, dan y_pred yaitu nilai prediksinya terdahap $i$ sebagai urutan data dalam dataset.

   Hasil nilai RMSE yang rendah menunjukkan bahwa variasi nilai yang dihasilkan dari model sistem rekomendasi mendekati variasi nilai observasinya. Artinya, semakin kecil nilai RMSE, maka akan semakin dekat nilai yang diprediksi dan diamati.

   Berikut merupakan visualisasi hasil training dan validation error dari metrik RMSE serta training dan validation loss ke dalam grafik plot.

   <img width="781" alt="image" src="https://github.com/user-attachments/assets/a0b32bd8-c2d1-48f0-baf6-f4bfcd182752" />

## Kesimpulan
Kesimpulannya adalah model yang digunakan untuk melakukan rekomendasi produk berdasarkan teknik Content-based Recommendation dan teknik Collaborative Filtering Recommendation telah berhasil dibuat dan sesuai dengan preferensi pengguna. Pada collaborative filtering diperlukan data rating dari pengguna, sedangkan pada content-based filtering, data rating tidak diperlukan karena analisis sistem rekomendasi akan berdasarkan atribut item dari masing-masing produk.

Referensi :

[1] Bawden, D., & Robinson, L. (2009). The dark side of information: Overload, anxiety and other paradoxes and pathologies. Journal of Information Science, 35(2), 180–191.

[2] Gomez-Uribe, C. A., & Hunt, N. (2016). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 1–19.

[3] McKinsey & Company. (2013). Big data, the next frontier for innovation, competition, and productivity.

[4] Jannach, D., Adomavicius, G., Tuzhilin, A., et al. (2016). Recommender Systems—Challenges, Insights and Research Opportunities. ACM Transactions on Intelligent Systems and Technology, 7(1), 1–34.

[5] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0

[6] Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.

[7] Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep Learning based Recommender System: A Survey and New Perspectives. ACM Computing Surveys (CSUR), 52(1), 1–38.


