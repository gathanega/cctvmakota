# Laporan CCTVMAKOTA — Deteksi Kendaraan & Rekomendasi Rute Kota Malang

## 1. Latar belakang

Kepadatan di simpang sekitar Universitas Brawijaya (Dinoyo, Sigura-Gura, Soekarno Hatta, Veteran, MT Haryono) sering berubah cepat. CCTVMAKOTA memakai 5 CCTV ATCS Kota Malang. Sistem menghitung kendaraan secara otomatis dengan model YOLO11s, menentukan status **LANCAR / PADAT / MACET**, lalu menyarankan jalur alternatif bila jalur utama macet.

## 2. Arsitektur

```
CCTV ATCS (stream) ──setiap 30 dtk──> ffmpeg (ambil 1 frame)
   └─> vision.detect_vehicles  (CLAHE otomatis → YOLO frame penuh + 6 tile → gabung NMS)
        └─> filter ROI (titik bawah-tengah kotak di dalam poligon jalur)
             └─> jumlah per kelas → status (ambang padat/macet per lokasi)
                  └─> atcs_routing: V/C ratio + BPR → waktu tempuh → rute utama / alternatif
                       └─> API FastAPI (app.py) → dashboard (dashboard.html) + Supabase (riwayat, opsional)
```

| Komponen | Berkas |
|---|---|
| Server, polling, API | `app.py`, `frame_queue.py` |
| Deteksi | `vision.py` |
| Routing | `atcs_routing.py`, `demo_macet.py` |
| GIS OSM | `gis_malang.py`, `gis/` |
| Admin | `admin_api.py`, `runtime.py` |
| Dashboard | `cctvmakota-dashboard/index.html` → `dashboard.html` |

Rumus waktu tempuh (BPR): `t = t0 × (1 + 0,15 × (V/C)^4)`. Rute alternatif dipilih jika lebih cepat dari rute utama yang macet.

## 3. Tata cara penggunaan

### 3.1 Menjalankan
```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8011
```
Buka `http://127.0.0.1:8011/`. Dari HP pakai `http://<IP-komputer>:8011/`.

### 3.2 Dashboard
Menampilkan status lokasi terpilih, jumlah kendaraan di ROI dan di semua jalur, tingkat kepadatan dengan ambang, komposisi jenis kendaraan, rata-rata per jam, dan data terbaru. Lokasi dipilih dari menu di header.

| Mobile | Tablet | Windows |
|---|---|---|
| <img src="screenshots/m_dashboard.jpg" width="200"> | <img src="screenshots/t_dashboard.jpg" width="260"> | <img src="screenshots/w_dashboard.jpg" width="300"> |

### 3.3 Pantauan & Rute
Halaman ini berisi rekomendasi rute, peta pin CCTV berwarna sesuai status, analisis GIS, perbandingan rute dengan simulasi kepadatan, kamera live dengan ROI, dan tabel hasil deteksi (unduh JPG/PNG/JSON/CSV).

| Mobile | Tablet | Windows |
|---|---|---|
| <img src="screenshots/m_pantauan.jpg" width="200"> | <img src="screenshots/t_pantauan.jpg" width="260"> | <img src="screenshots/w_pantauan.jpg" width="300"> |

### 3.4 Navigasi (demo kemacetan A → B)
1. Buka tab **Navigasi**, lalu izinkan lokasi atau tekan **"Coba dengan titik demo"**.
2. Pilih tingkat simulasi **Lancar 0% / Padat 60% / Macet 95%** atau geser slider.
3. Pada 0% dan 60%, jalur utama Gajayana tetap dipakai. Pada 95% (MACET), Gajayana butuh ±7,7 menit, jadi sistem beralih ke **Alternatif 1 Sunan Kalijaga** (±4,2 menit, hemat ±3,5 menit).
4. Lihat petunjuk arah atau tekan **Buka di Google Maps**.

| Izin lokasi | Lancar 0% | Padat 60% | Macet 95% |
|---|---|---|---|
| ![](screenshots/w_demo_1_izin_lokasi.jpg) | ![](screenshots/w_demo_2_lancar_0.jpg) | ![](screenshots/w_demo_3_padat_60.jpg) | ![](screenshots/w_demo_4_macet_95.jpg) |
| ![](screenshots/m_demo_1_izin_lokasi.jpg) | ![](screenshots/m_demo_2_lancar_0.jpg) | ![](screenshots/m_demo_3_padat_60.jpg) | ![](screenshots/m_demo_4_macet_95.jpg) |

### 3.5 Riwayat
Menampilkan tabel riwayat deteksi dengan tombol muat ulang dan unduh CSV. Halaman ini butuh Supabase (`SUPABASE_SERVICE_KEY`). Tanpa itu, tabelnya kosong seperti di screenshot.

| Mobile | Tablet | Windows |
|---|---|---|
| <img src="screenshots/m_riwayat.jpg" width="200"> | <img src="screenshots/t_riwayat.jpg" width="260"> | <img src="screenshots/w_riwayat.jpg" width="300"> |

### 3.6 Admin
Buka `#admin` dan masukkan token admin. Halaman ini berisi:
- **Ringkasan**: status seluruh kamera.
- **Kontrol rute**: memaksa rute utama atau alternatif, atau kembali ke otomatis.
- **Ruas jalan**: kapasitas dan panjang ruas, dengan gambar ruas di peta.
- **Lokasi & ambang**: menyeret pin kamera dan mengatur ambang padat/macet.
- **Editor ROI**: menggambar poligon jalur di atas frame kamera, lalu menyimpannya.

| Mobile | Tablet | Windows |
|---|---|---|
| <img src="screenshots/m_admin.jpg" width="200"> | <img src="screenshots/t_admin.jpg" width="260"> | <img src="screenshots/w_admin.jpg" width="300"> |

## 4. Model deteksi kendaraan

### 4.1 Dataset
Frame diambil langsung dari 5 CCTV Malang (`training/collect_malang_dataset_smart.py`), ditambah frame malam dan hujan (`add_condition_frames.py`). Semua frame diberi pra-label dengan yolo11x (`prepare_malang_labels.py`), lalu dikoreksi manual.

| Split | Gambar | car | motorcycle | bus | truck | Total kotak |
|---|---|---|---|---|---|---|
| train | 584 | 2080 | 880 | 20 | 664 | 3644 |
| val | 143 | 509 | 176 | 5 | 141 | 831 |

![Distribusi label](model/labels.jpg)

### 4.2 Training
- Base model: `yolo11s.pt`, fine-tune dengan `training/train_malang.py`.
- Parameter: epochs 100 (patience 15), batch 16, imgsz 960, lr0 0.01, hsv_v 0.6 (variasi terang/gelap), fliplr 0.5, mosaic 1.0, GPU.
- Training berhenti otomatis di epoch 77. **Epoch terbaik adalah 62**. Lama training 17,8 menit.
- Hasil: `runs/yolo11s_malang_live/weights/best.pt` (18 MB). Ringkasan lengkap ada di [model/train_summary.json](model/train_summary.json).

![Kurva training](model/results.png)

### 4.3 Hasil validasi

| Kelas | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0,818 | 0,532 | 0,684 | 0,541 |
| motorcycle | 0,765 | 0,568 | 0,714 | 0,462 |
| bus | 0,000 | 0,000 | 0,006 | 0,002 |
| truck | 0,781 | 0,758 | 0,832 | 0,764 |
| **Semua** | **0,591** | **0,465** | **0,559** | **0,442** |

| Kurva PR | Kurva F1 | Confusion matrix |
|---|---|---|
| ![](model/BoxPR_curve.png) | ![](model/BoxF1_curve.png) | ![](model/confusion_matrix_normalized.png) |

| Label asli (val) | Prediksi model |
|---|---|
| ![](model/val_batch0_labels.jpg) | ![](model/val_batch0_pred.jpg) |

**Uji live** pada 200 frame menghasilkan 1472 deteksi (rata-rata 7,36 per frame) dengan kecepatan 12,4 ms/frame: car 756, motorcycle 405, bus 8, truck 303.

**Benchmark model COCO** pada data yang sama ([model/benchmark_coco.csv](model/benchmark_coco.csv)):

| Model | mAP50 | F1 | ms/frame |
|---|---|---|---|
| yolov8n | 0,258 | 0,40 | 8,1 |
| yolov8s | 0,317 | 0,47 | 9,4 |
| yolo11n | 0,275 | 0,43 | 8,5 |
| yolo11x | 0,569 | 0,71 | 42,0 |

Benchmark ini menguntungkan yolo11x, karena label datasetnya berasal dari pra-label yolo11x. Model fine-tune yolo11s dipilih karena akurasinya mendekati yolo11x tetapi ±3,4× lebih cepat. Percobaan retrain dengan label ulang ditolak (`best_relabel_ditolak.pt`) karena hasilnya lebih buruk.

### 4.4 Cara model menentukan deteksi (`vision.py`)
1. **Penajaman (enhance: auto):** jika rata-rata kecerahan frame < 90 (malam) atau kontrasnya std < 35 (hujan/kabut), sistem menerapkan CLAHE agar kendaraan lebih jelas.
2. **Inferensi frame penuh** pada imgsz 960.
3. **Tiling 2×3:** frame dipotong menjadi 6 tile dengan overlap 25%, lalu tiap tile diinferensi pada ukuran 640. Langkah ini membantu mendeteksi kendaraan kecil atau jauh.
4. **Penggabungan (`merge_boxes`):** kotak dari frame penuh dan tile digabung dengan NMS (iou 0.65). Kotak duplikat atau kotak yang terpotong batas tile disatukan, dan yang dipertahankan adalah kotak dengan keyakinan tertinggi.
5. **Filter keyakinan:** hanya kotak dengan conf ≥ 0.25 dan kelas kendaraan (car/motorcycle/bus/truck) yang dipakai.
6. **Filter ROI:** titik bawah-tengah setiap kotak diuji dengan `_point_in_polygon`. Hanya kendaraan di dalam jalur yang dihitung.
7. **Status:** jumlah kendaraan di ROI dibandingkan dengan ambang padat/macet tiap lokasi. Hasilnya menjadi masukan V/C untuk routing.

Rata-rata waktu proses ±55 ms/frame dengan tiling.

### 4.5 Contoh hasil deteksi

| Soekarno Hatta (42) | Veteran (15) | MT Haryono (11) |
|---|---|---|
| ![](screenshots/deteksi_JlSoekarnoHattaArahSelatan.jpg) | ![](screenshots/deteksi_JlVeteranArahTimur.jpg) | ![](screenshots/deteksi_JlMtHaryonoArahBarat.jpg) |

### 4.6 Dataset yang dipilih: `dataset_malang_labeled`

| Keterangan | Nilai |
|---|---|
| Nama | `dataset_malang_labeled` (CCTV Malang, label terkoreksi) |
| Path | `E:\Gathan folder\flowincheck-main\dataset_malang_labeled` |
| Config | `dataset_malang_labeled/data.yaml` (nc 4: car, motorcycle, bus, truck) |

**Data training**

| Item | Train | Val |
|---|---|---|
| Total gambar | 584 | 143 |
| Total instance | 3.644 | 831 |
| car | 2.080 | 509 |
| motorcycle | 880 | 176 |
| bus | 20 | 5 |
| truck | 664 | 141 |

- Semua gambar adalah frame asli dari 5 CCTV ATCS Malang (pagi, siang, malam, hujan), tanpa gambar sintetis. 30 gambar train berasal dari frame tambahan malam/hujan.
- Labelnya dibuat dulu oleh yolo11x, lalu dikoreksi manual (700 file di folder `review/`). Tidak ada gambar yang labelnya hilang atau kosong.
- Imbalance ratio (kelas terbanyak / tersedikit) adalah 2.080 / 20 = **104**. Jika bus tidak dihitung: 2.080 / 664 ≈ **3,1**.
- Ukuran penyimpanan ±199 MB.

**Performa model (yolo11s, validasi 143 gambar)**

| Metrik | Nilai |
|---|---|
| mAP50 | 0,559 |
| mAP50-95 | 0,442 |
| AP50 motorcycle | 0,714 (recall 56,8%) |
| AP50 car / truck | 0,684 / 0,832 |
| AP50 bus | 0,006 |
| Waktu training | 17,8 menit (berhenti otomatis di epoch 77, terbaik di epoch 62) |
| Kecepatan | 12,4 ms/frame (±55 ms dengan tiling) |

**Alasan pemilihan**
1. Dataset ini sesuai dengan kondisi di lapangan, karena seluruh gambarnya berasal dari kamera yang benar-benar dipakai (sudut, resolusi, dan cahaya sama).
2. Labelnya sudah dikoreksi manual, sehingga lebih bisa dipercaya daripada label otomatis.
3. Hasilnya bagus untuk kelas utama: mAP50 truck 0,832, motorcycle 0,714, dan car 0,684. Bandingkan dengan model COCO yolo11s/yolov8s yang hanya mencapai mAP50 ±0,3.
4. Training cepat (17,8 menit) dan ukurannya kecil (199 MB), sehingga mudah dilatih ulang.
5. Model hasilnya ringan (18 MB) dan cukup cepat untuk polling CCTV setiap 30 detik.

**Kelemahan:** kelas bus sangat kurang (20 instance), sehingga bus hampir tidak terdeteksi. Recall secara umum juga masih 0,465. Perbaikan berikutnya adalah menambah frame berisi bus dan kendaraan kecil/jauh, lalu melatih ulang.
## 5. Keterbatasan
- **Bus hampir tidak terdeteksi** (AP ≈ 0) karena hanya ada 20 kotak bus di data latih. Bus cenderung terbaca sebagai truck.
- **Recall 0,465:** kendaraan yang tertutup atau sangat kecil masih terlewat. Tiling mengurangi masalah ini tetapi tidak menghilangkannya.
- Label validasi berasal dari pra-label yolo11x yang dikoreksi, sehingga metrik bisa bias.
- Nilai V/C dan kapasitas ruas masih perkiraan, belum dikalibrasi dengan survei lapangan.
- GPS di HP butuh HTTPS.

## 6. Pengembangan lanjut
- Tambah data bus dan frame malam/hujan, lalu latih ulang.
- Kalibrasi kapasitas ruas dengan data lapangan.
- Pasang HTTPS dan simpan token admin di environment variable.
