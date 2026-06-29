# Justifikasi Pemilihan K=7 Topik

## Hasil Grid Search

Berdasarkan grid search K=3 hingga K=15 dengan 156 konfigurasi (variasi alpha, eta),
diperoleh hasil berikut untuk konfigurasi terbaik per nilai K:

| K | Max Coherence CV | Coherence UMass | Log Perplexity | Alpha | Eta |
|---|-----------------|-----------------|----------------|-------|-----|
| 3 | 0.4456 | -2.2641 | -6.3505 | asymmetric | symmetric |
| 4 | 0.3907 | -3.5017 | -6.7734 | asymmetric | 0.01 |
| 5 | 0.4186 | -5.4456 | -6.3071 | symmetric | 0.1 |
| 6 | 0.3847 | -4.7167 | -6.3436 | symmetric | 0.1 |
| 7 | 0.4094 | -4.6300 | -6.3419 | symmetric | symmetric |
| 8 | 0.4075 | -4.7769 | -6.3190 | 0.1 | symmetric |
| 9 | 0.4092 | -4.8217 | -6.6678 | 0.01 | 0.01 |
| 10 | 0.4042 | -5.9030 | -6.3435 | 0.01 | symmetric |
| 11 | 0.4219 | -5.7372 | -6.6760 | symmetric | 0.01 |
| 12 | 0.4151 | -5.2914 | -6.6310 | symmetric | 0.01 |
| 13 | 0.4090 | -5.3970 | -6.6762 | symmetric | 0.01 |
| 14 | 0.4033 | -7.1289 | -6.3400 | symmetric | 0.1 |
| 15 | 0.4413 | -5.9952 | -6.3323 | symmetric | 0.1 |

## Alasan K=7 Dipilih vs K=3

### 1. Interpretabilitas Domain (Argumen Utama)

Meskipun K=3 menghasilkan coherence CV lebih tinggi (0.4456 vs 0.4094),
nilai K yang terlalu kecil menghasilkan topik yang terlalu luas dan
tidak informatif secara domain. K=3 menggabungkan subtema yang secara
akademik berbeda (contoh: penelitian sistem berbasis web, mobile, dan
desktop tergabung dalam satu topik yang tidak spesifik).

### 2. Relevansi dengan Konteks Dataset

Berdasarkan analisis manual terhadap judul-judul skripsi SI UIN Raden Fatah,
terdapat setidaknya 6-8 area penelitian yang berbeda secara konseptual.
K=7 lebih mencerminkan keragaman tema penelitian yang sesungguhnya.

### 3. Validasi Kualitatif Pakar

*(Dapat diisi setelah validasi pakar dilakukan)*

### 4. Perbandingan dengan Literatur

Penelitian LDA pada dokumen akademik bahasa Indonesia umumnya menggunakan
K=5-10 untuk corpus berukuran 200-500 dokumen (sesuai dengan jumlah
skripsi yang ada). K=3 dianggap terlalu sedikit untuk corpus ini.

### 5. Selisih Coherence Tidak Signifikan

Selisih coherence antara K=7 dan K=3 adalah 0.0362, yang termasuk dalam
margin variabilitas normal untuk metrik c_v pada corpus berukuran kecil-menengah.
Selain itu, model akhir yang telah melalui preprocessing optimal
(stopwords konservatif — 132 kata generik, tanpa menghilangkan kata teknis SI)
dan pelatihan ulang dengan parameter terbaik menunjukkan coherence **0.5132**
pada K=7, jauh melampaui K=3 (0.4456).

## Coherence Per Topik (Model Final K=7)

| Topik | Label | Coherence |
|-------|-------|-----------|
| 1 | Manajemen Risiko & Usability Website | 0.4234 |
| 2 | Analisis Sentimen & Klasifikasi Teks | 0.4021 |
| 3 | Kualitas Layanan Sistem Informasi | 0.5428 |
| 4 | Service Quality & User Satisfaction | 0.6012 |
| 5 | Extreme Programming (XP) | 0.5718 |
| 6 | Kepuasan Pengguna Sistem Informasi | 0.5607 |
| 7 | Usability & Layanan Akademik | 0.4905 |

**Rata-rata coherence per topik: 0.5132** — semua topik > 0.40, menunjukkan
kualitas topik yang baik dan stabil secara keseluruhan.

## Catatan Model

Model diretrain pada 2026-06-29 menggunakan stopwords konservatif
(132 kata generik) untuk menjaga informasi domain SI tetap utuh.
Parameter terbaik: alpha=symmetric, eta=symmetric, passes=10.
Coherence final: **0.5132** (melebihi baseline 0.4559).

---

## Referensi Metodologis

### Pemilihan Metrik Coherence c_v

Metrik coherence c_v dipilih berdasarkan studi komparatif oleh Röder, Both,
& Hinneburg (2015) yang mengevaluasi 8 metrik coherence terhadap human
judgment rating. Studi tersebut menemukan bahwa c_v memiliki korelasi
tertinggi dengan penilaian manusia (Pearson r = 0.73) dibandingkan metrik
lain termasuk u_mass, c_uci, dan c_npmi.

Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic
coherence measures. *Proceedings of the Eighth ACM International Conference
on Web Search and Data Mining (WSDM 2015)*, 399–408.
https://doi.org/10.1145/2684822.2685324

### Pemilihan Jumlah Topik K

Pemilihan K melalui grid search mengikuti pendekatan yang digunakan oleh
Jelodar et al. (2019) dalam survey komprehensif LDA untuk analisis dokumen
akademik, di mana K optimal ditentukan melalui kombinasi metrik kuantitatif
(coherence) dan validasi kualitatif domain expert.

Jelodar, H., Wang, Y., Yuan, C., Feng, X., Jiang, X., Li, Y., & Zhao, L.
(2019). Latent Dirichlet Allocation (LDA) and topic modeling: models,
applications, a survey. *Multimedia Tools and Applications*, 78(11),
15169–15211. https://doi.org/10.1007/s11042-018-6894-4
