# Fitur Dashboard & Admin (screenshot halaman penuh)

Setiap gambar adalah **screenshot panjang satu halaman penuh**, jadi semua fitur terlihat dari atas sampai bawah tanpa terpotong. Klik gambar untuk melihat ukuran asli.

- **Mobile**: lebar 390 px. Tabel ditampilkan bertumpuk (label : nilai) agar tidak terpotong ke samping.
- **Tablet**: lebar 820 px.
- **Windows**: desktop.

## Ringkasan fitur per halaman

| Halaman | Fitur |
|---|---|
| Dashboard | Status lalu lintas, kendaraan di jalur ROI, semua jalur, jam & tanggal, tingkat kepadatan + ambang padat/macet, jenis kendaraan, rata-rata per jam, daftar lokasi, data terbaru |
| Pantauan & Rute | Rekomendasi rute, peta pin CCTV berwarna status, daftar titik, analisis GIS jalan Malang (OSM), perbandingan rute + simulasi kepadatan, kamera live + ROI, hasil deteksi ROI (unduh JPG/PNG/JSON/CSV) |
| Navigasi | Izin lokasi, "Anda sekarang di", peringatan berwarna, peta rute A→B, simulasi kemacetan (Lancar/Padat/Macet), pilihan jalur, petunjuk arah, buka di Google Maps |
| Riwayat | Tabel riwayat deteksi, jumlah baris, muat ulang, unduh CSV |
| Admin | Ringkasan, Kontrol rute, Ruas jalan (+ gambar ruas di peta), Lokasi & ambang (seret pin kamera), Editor ROI |

## Dashboard
| Mobile | Tablet | Windows |
|---|---|---|
| ![](screenshots/m_dashboard.jpg) | ![](screenshots/t_dashboard.jpg) | ![](screenshots/w_dashboard.jpg) |

## Pantauan & Rute
| Mobile | Tablet | Windows |
|---|---|---|
| ![](screenshots/m_pantauan.jpg) | ![](screenshots/t_pantauan.jpg) | ![](screenshots/w_pantauan.jpg) |

## Navigasi (tes kemacetan 95%)
| Mobile | Tablet | Windows |
|---|---|---|
| ![](screenshots/m_demo_4_macet_95.jpg) | ![](screenshots/t_demo_4_macet_95.jpg) | ![](screenshots/w_demo_4_macet_95.jpg) |

Tahap lain (izin lokasi, 0%, 60%) ada di [README.md](README.md#5-screenshot).

## Riwayat
Supabase belum dikonfigurasi, jadi tabel masih kosong.

| Mobile | Tablet | Windows |
|---|---|---|
| ![](screenshots/m_riwayat.jpg) | ![](screenshots/t_riwayat.jpg) | ![](screenshots/w_riwayat.jpg) |

## Admin (semua bagian dalam satu halaman)
Login dengan token admin. Urutan dari atas: Ringkasan → Kontrol rute → Ruas jalan (+ peta) → Lokasi & ambang (+ peta & tabel) → Editor ROI.

| Mobile | Tablet | Windows |
|---|---|---|
| ![](screenshots/m_admin.jpg) | ![](screenshots/t_admin.jpg) | ![](screenshots/w_admin.jpg) |