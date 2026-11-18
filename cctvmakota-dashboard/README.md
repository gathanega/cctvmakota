# Dashboard ATCS — CCTVMAKOTA

Folder khusus untuk seluruh berkas UI dashboard. Tidak ada logika backend di sini;
server yang menyajikan halaman berada di `../atcs_unified/app.py`.

## Isi folder

| Berkas | Rute | Keterangan |
|--------|------|------------|
| `dashboard.html` | `/` | **Satu file mandiri** dashboard CCTVMAKOTA (CSS, JS, logo inline) |
| `index.html` | – | Sumber editable (logo dari `assets/`) |
| `build_single_file.py` | – | Membangun ulang `dashboard.html` dari `index.html` |
| `assets/cctvmakota-logo.svg` | – | Logo CCTVMAKOTA (juga favicon) |

`dashboard.html` sudah memuat logo sebagai data URI, jadi bisa dibuka langsung
(file://) atau dikirim sebagai satu berkas tanpa kehilangan gambar.

## Mengubah dashboard

1. Edit `index.html` (versi mudah dibaca).
2. Jalankan:
   ```powershell
   cd "C:\Users\Than\Documents\Default Project\CCTVMAKOTA\dashboard"
   python build_single_file.py
   ```
3. `dashboard.html` akan diperbarui otomatis.

## Sumber data

Dashboard memanggil API di server yang sama:

- `/api/public-config`
- `/api/locations`
- `/api/logs`
- `/api/hourly`
- `/api/insight?location=...`
- `/api/frame` dan `/api/detections` (opsional, 404 bila belum ada frame)

Data diambil dari SQLite `../atcs_unified/data/atcs.sqlite3`.

## Menjalankan

```powershell
$env:PYTHONPATH="C:\Users\Than\AppData\Local\Temp\opencode\fic\pylib"
cd "C:\Users\Than\Documents\Default Project\CCTVMAKOTA\atcs_unified"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Buka:

- http://127.0.0.1:8000/ — dashboard CCTVMAKOTA

## Catatan

- Tema default terang; tombol tema di header bisa mengubah ke gelap.
- Library eksternal tetap dari CDN: Google Fonts (Inter), Chart.js, Leaflet,
  hls.js. Tanpa internet, chart/peta tidak tampil.
- Data diperbarui lewat HTTP polling API (tanpa MQTT).
- Untuk menghentikan data simulasi:
  ```sql
  DELETE FROM observations WHERE source='demo-seed';
  ```
