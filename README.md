# 👹FaceDig TPS — Analisis Geometrik Morfometrik Wajah👹

Analisis variasi bentuk wajah laki-laki dan perempuan menggunakan **geometric morphometrics** dari data landmark FaceDig AI.

## 📋 Deskripsi

Project ini menganalisis perbedaan morfologi wajah berdasarkan **72 landmark** yang dihasilkan oleh [FaceDig AI](https://facedig.com) dari foto formal. Data mencakup 3 angkatan tahun (2023, 2024, 2025) dengan total 61 specimen.

### Pipeline Analisis

```
Foto Formal → FaceDig AI → File .TPS (72 landmark)
   │
   ▼
Permutasi → TPS Terpisah (2023, 2024, 2025)
             │
             ▼
Visualisasi TPS (Python script) → Gambar 
             │
             ▼
Parsing TPS → GPA (Procrustes Alignment) → PCA → CVA → PCoA
             │
             ▼
Uji Statistik (Permutation MANOVA, t-test, Kruskal-Wallis)
             │
             ▼
Visualisasi + Export CSV
```
### Visualisasi TPS
⬛ Dilakukan dengan cara buka terminal di folder yang sama dengan file VisualisasiTPS.py 
kemudian ketik perintah salah satu dibawah ini (sesuaikan):
#### Batch Processing TPS Files
```bash
python VisualisasiTPS.py
```
#### Individual TPS File
```bash
python VisualisasiTPS.py --tps Data/2023/2023_AI.tps --out-dir "Data/2023/Output 2023"
#python VisualisasiTPS.py --tps Data/2024/2024_AI.tps --out-dir "Data/2024/Output 2024"
#python VisualisasiTPS.py --tps Data/2025/2025_AI.tps --out-dir "Data/2025/Output 2025"
```
aktifkan salah satunya dengan hilangkan tanda #

hasil dari visualisasinya akan ditambilkan didalam folder yang sama dengan folder tps yang diinputkan misalnya pada contoh diatas maka hasil visualisasinya akan ditambilkan didalam folder `Data/2023/Output 2023`

## 📁 Struktur Folder

```
facedig-project/
├── data/
│   ├── 2023/                   # 10 L + 10 P + 2023_AI.tps
│   │   ├── Output 2023/        # Hasil visualisasi TPS
│   ├── 2024/                   # 10 L + 10 P + 2024_AI.tps
│   │   ├── Output 2024/        # Hasil visualisasi TPS
│   ├── 2025/                   # 10 L + 11 P + 2025_AI.tps
│   │   ├── Output 2025/        # Hasil visualisasi TPS
├── output/
│   ├── figures/                # Semua plot PNG
│   └── tables/                 # CSV statistik
├── facedig_analisis.ipynb      # Notebook utama
├── VisualisasiTPS.py           # Visualisasi TPS
├── environment.yml             # Conda environment
└── README.md                   # README.md
```

**Format penamaan foto:**
- `L_` = Laki-laki (contoh: `L_Angkatan_NomorProbandus_SukuAsal.jpg`)
- `P_` = Perempuan (contoh: `P_Angkatan_NomorProbandus_SukuAsal.jpg`)

## ⚙️ Setup & Instalasi Analisis ipynb  

### 1. Buat Conda Environment

```bash
conda env list
conda env create -f environment.yml
conda activate facedig-morpho
```

### 2. Jalankan Notebook

```bash
jupyter notebook facedig_analisis.ipynb
```

## 📊 Metode Analisis

| Metode | Tujuan |
|--------|--------|
| **GPA** (Generalized Procrustes Analysis) | Normalisasi ukuran, posisi, dan rotasi landmark |
| **PCA** (Principal Component Analysis) | Mereduksi dimensi, melihat pola variasi bentuk keseluruhan |
| **CVA** (Canonical Variate Analysis) | Memaksimalkan pemisahan antar-grup (L vs P) |
| **PCoA** (Principal Coordinate Analysis) | Ordination berbasis Procrustes distance matrix |
| **Permutation MANOVA** | Uji signifikansi efek jenis kelamin & tahun |
| **t-test / Kruskal-Wallis** | Uji perbedaan PC scores & centroid size |

## 📈 Output yang Dihasilkan

### Figures (`output/figures/`)

| File | Deskripsi |
|------|-----------|
| `01_landmark_foto.png` | Visualisasi 72 landmark di atas foto contoh |
| `02_mean_shape.png` | Perbandingan mean shape L vs P per tahun |
| `03_pca_sex.png` | PCA scatter + 95% confidence ellipse (L vs P) |
| `04_pca_year_sex.png` | PCA per tahun & jenis kelamin |
| `05_stat_plots.png` | Boxplot, violin plot, errorbar statistik |
| `06_cva.png` | CVA histogram (L vs P) + akurasi klasifikasi |
| `07_pcoa.png` | PCoA berdasarkan Procrustes distance |
| `08_heatmap.png` | Heatmap perbedaan landmark L vs P |

### Tables (`output/tables/`)

| File | Deskripsi |
|------|-----------|
| `deskriptif_statistik.csv` | Mean, SD per grup (tahun × sex) |
| `pca_variance.csv` | Varians tiap PC |
| `pc_scores.csv` | Skor PC1-PC3 + centroid size per specimen |

## 🔧 Dependencies

- Python 3.10
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (PCA, LDA)
- SciPy (statistik)
- Pillow (image processing)

## 📚 Referensi

- Adams, D.C., Rohlf, F.J., & Slice, D.E. (2013). A field comes of age: geometric morphometrics in the 21st century. *Hystrix*, 24(1), 7-14.
- Bookstein, F.L. (1991). *Morphometric Tools for Landmark Data*. Cambridge University Press.
- Rohlf, F.J. (2015). The tps series of software. *Hystrix*, 26(1), 9-12.
- Klingenberg, C.P. (2011). MorphoJ: an integrated software package for geometric morphometrics. *Molecular Ecology Resources*, 11(2), 353-357.

## 📝 Lisensi

Project ini untuk keperluan akademik/riset.
# FaceDig-Analysis
