# MELBA Phase 2 Integration Audit

**Tanggal audit:** 26 November 2026  
**Cakupan:** Regression Engine v1 dan Economic Mathematics Engine Phase 2  
**Status perubahan:** Audit dan dokumentasi saja; kode produk tidak diubah selama audit ini.

## Ringkasan

API regresi dan ekonomi terpisah dengan jelas dan dapat digunakan bersama dari source tree. Perhitungan ekonomi dasar dan turunan polinomial memakai rumus yang sesuai. Optimasi keuntungan mengevaluasi batas interval dan titik stasioner, lalu memeriksa turunan kedua sebelum memberi label maksimum atau minimum. Alur demonstrasi regresi-ke-profit berhasil.

Semua 91 tes yang tersedia lulus. Tes notebook bootstrap termasuk di dalam hasil tersebut. Namun, instalasi dari artefak wheel yang tersedia tidak menyediakan melba.economics. Wheel itu bertanggal 25 November dan tidak memuat modul Phase 2. Build wheel baru tidak dapat diverifikasi karena pip membutuhkan setuptools>=77.0.3 dan akses jaringan lingkungan ditolak. Karena itu, hasil source tree dan notebook tidak membuktikan bahwa distribusi package Phase 2 sudah siap.

**Rekomendasi:** tahan rilis/distribusi package sampai wheel baru berhasil dibangun, diinstal secara terisolasi, dan impor semua modul ekonomi berhasil. Phase 3 dapat direncanakan setelah temuan integrasi dan satuan di bawah ditinjau; jangan menganggap instalasi package sebagai terverifikasi.

## Metode dan artefak yang diperiksa

Dokumen dan kode yang diperiksa:

- docs/audit-v1.md
- docs/economics.md
- README.md
- melba/economics/__init__.py
- melba/economics/_validation.py
- melba/economics/linear.py
- melba/economics/differential.py
- melba/economics/elasticity.py
- API model dan validasi regresi yang digunakan dalam alur integrasi
- pyproject.toml, contoh, dan pemeriksaan bootstrap notebook

Instruksi repository menyarankan penggunaan skill codebase-memory untuk eksplorasi struktur. File skill tidak dapat dibaca pada lingkungan audit ini, sehingga relasi package dan API diperiksa langsung dari file repository. Kesimpulan audit tidak mengandalkan hasil graph.

## Hasil audit struktur dan API

### Package dan pemakaian

- Engine regresi tetap tersedia melalui API v1 (MELBA dari package melba). Engine ekonomi berada di melba.economics dan dapat diimpor tanpa mengubah namespace v1, misalnya: from melba.economics import RevenueFunction.
- Package discovery di pyproject.toml menggunakan include = ["melba*"]; secara konfigurasi, subpackage melba.economics termasuk dalam pencarian package.
- Subpackage ekonomi mempunyai __init__.py dan mengekspor API ekonominya. Struktur ini mudah ditemukan dan tidak menambahkan dependency baru.
- Pemisahan ini mempertahankan API regresi yang ada. Alur integrasi menghubungkan hasil prediksi dan fungsi ekonomi pada kode pemanggil, bukan mengubah optimizer regresi.

### Instalasi distribusi

Artefak lokal dist/melba-0.1.0-py3-none-any.whl berhasil dipasang ke direktori target sementara dengan pip install --no-deps. Artefak tersebut tidak memuat melba/economics/; percobaan import melba.economics dari instalasi itu gagal dengan ModuleNotFoundError.

Wheel lokal bertanggal 25 November 2026, sedangkan source tree audit memiliki modul ekonomi. Jadi wheel tersebut adalah artefak lama dan kegagalannya **tidak membuktikan** bahwa konfigurasi source saat ini mengecualikan subpackage. Build baru dicoba menggunakan python -m pip wheel --no-deps; build terisolasi gagal saat memasang build dependency karena setuptools>=77.0.3 tidak tersedia dan koneksi ke indeks paket ditolak oleh lingkungan. Build dan instalasi wheel Phase 2 saat ini belum terverifikasi.

Pemeriksaan notebook bootstrap yang menjadi bagian suite berhasil. Itu memvalidasi bootstrap dan sinkronisasi notebook yang diuji secara lokal; notebook tidak dijalankan pada layanan Google Colab.

## Hasil pemeriksaan rumus dan domain

| API | Rumus yang diterapkan | Catatan audit |
|---|---|---|
| DemandFunction | Qd = a - bP | Bentuk permintaan linear. Di luar harga choke, fungsi mengembalikan kuantitas negatif dari garis mentah; pemanggil harus membatasi analisis pada domain ekonomi yang layak. |
| SupplyFunction | Qs = c + dP | Bentuk penawaran linear. |
| CostFunction | TC = FC + vQ | v adalah biaya variabel per unit; biaya total untuk kuantitas nol tetap sama dengan biaya tetap. |
| RevenueFunction | TR = P × Q | Harga dan kuantitas harus ditafsirkan dalam satuan yang cocok. |
| ProfitFunction | π = TR - TC | Konsisten dengan pendapatan total dikurangi biaya total. |
| Turunan polinomial | Jika f(x) = Σ cₖxᵏ, maka f′(x) = Σ k cₖxᵏ⁻¹ | Koefisien disimpan dari pangkat terendah ke tertinggi; turunan dihitung secara analitis. |
| Marginal cost/revenue/profit | Turunan fungsi total terhadap variabel kuantitas yang diberikan | RevenueFunction menganggap harga tetap saat menghitung marginal revenue; ini sesuai d(PQ)/dQ = P hanya dengan asumsi tersebut. Untuk harga yang berubah terhadap kuantitas, gunakan polinomial total revenue yang merepresentasikan hubungan itu. |
| Elastisitas permintaan | Ed = (dQ/dP) × (P/Q) | Tanda elastisitas dipertahankan; kategori menggunakan nilai absolut: elastic jika |Ed| > 1, inelastic jika < 1, dan unitary elastic jika dekat dengan 1 dalam toleransi yang didokumentasikan. |

Optimasi keuntungan menerima interval batas yang diberikan pemanggil, menemukan kandidat akar turunan pertama, dan mengevaluasi kandidat serta kedua ujung interval. Label maksimum/minimum lokal memerlukan bukti dari turunan kedua; titik stasioner dengan klasifikasi turunan kedua yang tidak menentukan dilaporkan sebagai inconclusive, bukan otomatis maksimum. Ini mengikuti syarat matematis yang diminta.

## Scalar, array, dan validasi numerik

- Fungsi linear menerima scalar dan input berbentuk array; operasi array mengikuti broadcasting NumPy. Hasil scalar dikembalikan sebagai scalar Python, sedangkan input array menghasilkan numpy.ndarray.
- Input numerik non-finite (NaN dan infinity) ditolak oleh validasi. Fungsi yang memerlukan harga, kuantitas, biaya, atau pendapatan non-negatif menolak nilai negatif. Validasi hasil juga mendeteksi hasil non-finite/overflow dan mengeluarkan error alih-alih meneruskan angka tak hingga.
- Polinomial mengandalkan koefisien dan titik evaluasi numerik yang valid. Optimasi membatasi interval ke batas finite dan memeriksa bahwa batas terurut serta berada pada domain kuantitas non-negatif.
- Nilai sangat besar bisa menyebabkan overflow, yang ditolak; tidak ada jaminan akurasi floating point untuk koefisien atau skala yang sangat buruk kondisinya.
- Demand linear sengaja tetap mengembalikan nilai kuantitas negatif di luar domain garis yang layak. Fungsi permintaan matematis tidak otomatis memotong nilai pada nol.
- Pada elastisitas, P = 0 dan Q > 0 menghasilkan elastisitas nol sesuai rumus. Q <= 0 ditolak agar pembagi nol atau domain kuantitas tidak layak tidak disamarkan. Kuantitas positif yang sangat dekat nol dapat menghasilkan elastisitas sangat besar; selama masih finite, hasil tersebut dipertahankan. Nilai yang overflow ditolak. API tidak menetapkan ambang near-zero universal karena ambang itu bergantung pada satuan.

## Integrasi regresi dan ekonomi

Contoh ini memakai **data sintetis untuk demonstrasi API saja**. Fitur adalah anggaran promosi dalam ribuan unit mata uang, target adalah unit penjualan, dan semua asumsi harga/biaya dibuat untuk contoh. Contoh ini tidak mengukur akurasi dunia nyata dan tidak menyatakan bahwa anggaran promosi menyebabkan perubahan penjualan.

~~~python
import numpy as np
from melba import MELBA
from melba.economics import CostFunction, ProfitFunction, RevenueFunction

# Data sintetis: anggaran promosi (ribuan unit mata uang) -> unit terjual.
X = np.array([[0.], [10.], [20.], [30.], [40.], [50.]])
y = np.array([50., 70., 90., 110., 130., 150.])

model = MELBA(
    quadratic_terms=False,
    epochs=1000,
    random_state=42,
    l2_penalty=0.0,
)
model.fit(X, y)
predicted_sales = float(model.predict([[25.]])[0])

# Fungsi ekonomi mensyaratkan kuantitas non-negatif. Jangan diam-diam
# mengubah prediksi negatif; tangani sebagai prediksi di luar domain.
if predicted_sales < 0:
    raise ValueError(f"Prediksi penjualan di luar domain ekonomi: {predicted_sales}")

price_per_unit = 10_000.0
revenue = RevenueFunction().calculate(price_per_unit, predicted_sales)
total_cost = CostFunction(
    fixed_cost=100_000.0,
    variable_cost=6_000.0,
).calculate(predicted_sales)
profit = ProfitFunction().calculate(revenue, total_cost)
margin = profit / revenue if revenue else None

print(predicted_sales, revenue, total_cost, profit, margin)
~~~

Hasil yang diamati saat menjalankan contoh:

~~~text
predicted_sales=100.000000 units
revenue=1000000.00 currency units
total_cost=700000.00 currency units
profit=300000.00 currency units
profit_margin=0.300000
~~~

Perhitungan manual konsisten: TR = 10.000 × 100 = 1.000.000; TC = 100.000 + 6.000 × 100 = 700.000; π = 1.000.000 - 700.000 = 300.000; margin = 0,3. Prediksi positif pada contoh ini memungkinkan pemanggilan fungsi ekonomi. Prediksi regresi secara umum tidak dibatasi ke non-negatif, sehingga integrasi harus tetap memvalidasi hasil sebelum menghitung revenue/cost.

## Temuan dan prioritas

| Prioritas | Temuan | Dampak | Saran |
|---|---|---|---|
| **P1 — release gate** | Wheel lokal yang tersedia tidak menyertakan modul ekonomi, dan build wheel baru tidak dapat diuji karena build dependency tidak tersedia di lingkungan. | Pengguna yang memasang artefak wheel tersebut tidak dapat mengimpor API Phase 2. Belum ada bukti instalasi untuk distribusi Phase 2. | Bangun wheel dari source Phase 2 pada lingkungan dengan build dependency, instal ke environment/target bersih, lalu uji impor melba.economics dan submodulnya sebelum mendistribusikan. Jangan gunakan wheel lokal lama sebagai rilis Phase 2. |
| **P2 — integrasi/domain** | Output model regresi dapat negatif, sementara fungsi revenue dan cost menolak kuantitas negatif. | Prediksi tertentu tidak bisa langsung dipakai; tanpa pemeriksaan, pemanggilan ekonomi berakhir dengan error validasi. | Dokumentasikan pemeriksaan domain seperti pada contoh. Tentukan kebijakan bisnis untuk prediksi di luar domain pada layer pemanggil; jangan mengubah atau memotong hasil secara diam-diam. |
| **P2 — interpretasi** | API menerima angka tanpa metadata satuan. | API tidak dapat mendeteksi pencampuran mata uang, skala (misalnya rupiah vs ribuan rupiah), atau satuan kuantitas yang berbeda. | Pertahankan contoh dan dokumentasi yang menyebut satuan eksplisit. Pertimbangkan metadata satuan atau validasi domain pada tahap API mendatang tanpa menambah dependency berat. |
| **P3 — sensitivitas numerik** | Elastisitas untuk kuantitas positif yang sangat kecil dapat bernilai sangat besar; akar polinomial berderajat tinggi/berkondisi buruk juga rentan terhadap presisi floating point. | Nilai yang finite dapat sensitif terhadap perubahan kecil input/koefisien. | Terangkan rentang penggunaan dan tambahkan kasus uji numerik berkoefisien buruk sebelum memperluas penggunaan untuk analisis produksi. Hindari ambang kuantitas universal tanpa satuan. |

Tidak ditemukan rumus dasar yang salah pada fungsi yang diperiksa. Temuan terkait domain, skala, dan presisi perlu dipahami pemanggil, sementara temuan artefak instalasi perlu ditutup sebelum distribusi Phase 2 dinyatakan siap.

## Hasil pengujian

Perintah yang dijalankan:

~~~text
python -m pytest
~~~

Hasil:

~~~text
91 passed in 3.78s
Python 3.12.10 / pytest 9.1.1
~~~

Hasil ini mencakup suite regresi lama (48 tes baseline menurut audit v1), tes ekonomi, dan pemeriksaan notebook bootstrap. Tidak ada tes yang ditambahkan atau diubah selama audit ini.

Pemeriksaan distribusi:

- **Gagal/terblokir:** python -m pip wheel --no-deps ... tidak dapat memasang setuptools>=77.0.3 karena akses indeks paket ditolak. Wheel Phase 2 baru tidak terbentuk.
- **Berhasil untuk artefak lama:** instalasi sementara wheel lokal sukses, tetapi impor melba.economics gagal karena artefak itu tidak memuat subpackage ekonomi.
- **Notebook:** pemeriksaan bootstrap yang termasuk suite pytest lulus; eksekusi pada Google Colab tidak dilakukan.

## Daftar perbaikan yang disarankan

1. Rebuild dan verifikasi wheel Phase 2 pada lingkungan build yang mempunyai setuptools>=77.0.3; uji impor semua API ekonomi dari instalasi bersih.
2. Pertahankan validasi kuantitas hasil prediksi sebelum memasukkannya ke fungsi ekonomi, dan jelaskan bahwa model regresi tidak menjamin batas domain ekonomi.
3. Tambahkan contoh serta dokumentasi yang konsisten menyatakan mata uang, skala, periode, dan satuan kuantitas.
4. Tambahkan pengujian atau panduan untuk sensitivitas elastisitas dekat Q = 0 serta keterbatasan presisi optimasi polinomial berkondisi buruk.
5. Setelah review temuan ini, ulangi suite penuh dan pemeriksaan notebook/package pada kandidat distribusi yang akan dipakai.

## Kesiapan menuju Phase 3

**Source tree:** cukup siap untuk review Phase 2 dan perencanaan Phase 3; seluruh tes lulus dan contoh integrasi berhasil.  
**Distribusi package:** belum siap dinyatakan terverifikasi karena wheel lama tidak menyertakan ekonomi dan build wheel baru belum berhasil diuji.  
**Rekomendasi keseluruhan:** lanjutkan Phase 3 setelah laporan audit ini ditinjau, tetapi jadikan build serta instalasi wheel Phase 2 sebagai release gate sebelum pengguna mengandalkan instalasi package. Audit ini tidak mengimplementasikan Phase 3.
