# Impor library NumPy dan beri alias sebagai np
import numpy as np
# Impor modul cv2 untuk fungsi pengolahan citra
import cv2
# Impor modul pyplot dari Matplotlib dan beri alias sebagai plt
import matplotlib.pyplot as plt

# Buat array NumPy bernama img dengan nilai dan tipe data float32 yang telah ditentukan
img = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.float32)

# Tampilkan gambar asli menggunakan fungsi imshow dari Matplotlib
plt.imshow(img)

# Tentukan fungsi rekursif boundary_fill8 yang mengisi wilayah terhubung dalam gambar berdasarkan nilai batas
def boundary_fill8(img, col, row, fill, boundarrow):
    # Cetak nilai kolom dan baris saat ini untuk tujuan debugging
    print(f"kolom = {col}")
    print(f"baris = {row}")

    # Dapatkan nilai piksel saat ini
    current = img[col, row]

    # Jika piksel saat ini bukan nilai batas dan berbeda dari nilai pengisian, lakukan pengisian
    if current != boundarrow and np.float32(current) != np.float32(fill):
        img[col, row] = fill

        # Panggil fungsi secara rekursif dengan nilai kolom dan baris yang berbeda
        boundary_fill8(img, col + 1, row, fill, boundarrow)
        boundary_fill8(img, col - 1, row, fill, boundarrow)
        boundary_fill8(img, col, row + 1, fill, boundarrow)
        boundary_fill8(img, col, row - 1, fill, boundarrow)
        boundary_fill8(img, col + 1, row - 1, fill, boundarrow)
        boundary_fill8(img, col - 1, row - 1, fill, boundarrow)
        boundary_fill8(img, col + 1, row + 1, fill, boundarrow)
        boundary_fill8(img, col - 1, row + 1, fill, boundarrow)

# Buat salinan dari gambar asli untuk perbandingan
gambar_awal = np.array(img)

# Tampilkan gambar asli menggunakan fungsi imshow dari Matplotlib dengan peta warna 'jet' dan tambahkan judul
plt.imshow(gambar_awal, cmap='jet')
plt.title('Gambar Awal')
plt.show()

# Tentukan fungsi antialiasing yang akan memperlunak batas
# Tentukan fungsi supersampling untuk antialiasing
def supersampling_antialias(img, fill, boundarrow, alpha=0.5, num_samples=5):
    # Buat salinan gambar
    antialiased_img = np.copy(img)
    rows, cols = img.shape

    for col in range(cols):
        for row in range(rows):
            # Jika piksel saat ini adalah batas dan berbeda dari nilai pengisian,
            # gunakan supersampling untuk menghaluskan batas
            if img[col, row] == boundarrow and antialiased_img[col, row] != fill:
                subpixel_values = []

                for i in range(num_samples):
                    for j in range(num_samples):
                        subpixel_col = col + (i - num_samples // 2) / num_samples
                        subpixel_row = row + (j - num_samples // 2) / num_samples

                        if 0 <= subpixel_col < cols and 0 <= subpixel_row < rows:
                            subpixel_values.append(img[int(subpixel_col), int(subpixel_row)])

                # Hitung nilai rata-rata dari subpiksel
                averaged_value = np.mean(subpixel_values)
                antialiased_img[col, row] = fill + alpha * (averaged_value - fill)

    return antialiased_img

# Panggil fungsi boundary_fill8 untuk mengisi wilayah terhubung mulai dari kolom 12, baris 3, dengan nilai pengisian 0.6 dan nilai batas 1
boundary_fill8(img, 12, 3, 0.6, 1)

# Tambahkan antialiasing dengan supersampling ke gambar yang telah dimodifikasi
gambar_hasil = supersampling_antialias(img, 0.6, 1, num_samples=5)

# Tampilkan gambar yang telah dimodifikasi dengan supersampling antialiasing
plt.imshow(gambar_hasil, cmap='jet')
plt.title('Hasil warna wajah dengan Supersampling Antialiasing')
plt.show()

# Panggil fungsi boundary_fill8 untuk mengisi wilayah terhubung mulai dari kolom 6, baris 10, dengan nilai pengisian 0.85 dan nilai batas 1
boundary_fill8(img, 6, 10, 0.85, 1)

# Tambahkan antialiasing dengan supersampling ke gambar yang telah dimodifikasi
gambar_hasil = supersampling_antialias(img, 0.85, 1, num_samples=5)

# Tampilkan gambar yang telah dimodifikasi dengan supersampling antialiasing
plt.imshow(gambar_hasil, cmap='jet')
plt.title('Hasil warna rambut dengan Supersampling Antialiasing')
plt.show()  

# Fungsi translasi
# Fungsi translasi
def translate(img, tx, ty):
    # Mendapatkan jumlah baris dan kolom dari gambar
    rows, cols = img.shape

    # Membuat matriks translasi menggunakan NumPy
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    # Menggunakan cv2.warpAffine untuk menerapkan translasi
    translated_img = cv2.warpAffine(img, translation_matrix, (cols, rows))

    # Mengembalikan gambar yang telah diterjemahkan
    return translated_img

# Fungsi refleksi
def reflect(img, axis):
    # Memeriksa sumbu refleksi dan melakukan refleksi menggunakan NumPy
    if axis == 'x':
        reflected_img = np.flipud(img)
    elif axis == 'y':
        reflected_img = np.fliplr(img)

    # Mengembalikan gambar yang telah direfleksi
    return reflected_img

# Fungsi rotasi
def rotate(img, angle):
    # Mendapatkan jumlah baris dan kolom dari gambar
    rows, cols = img.shape

    # Membuat matriks rotasi menggunakan cv2.getRotationMatrix2D
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # Menggunakan cv2.warpAffine untuk menerapkan rotasi
    rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    # Mengembalikan gambar yang telah dirotasi
    return rotated_img

# Fungsi skala
def scale(img, scale_factor):
    # Mendapatkan jumlah baris dan kolom dari gambar
    rows, cols = img.shape

    # Menggunakan cv2.resize untuk menerapkan skala
    scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # Mengembalikan gambar yang telah diubah skala
    return scaled_img

# Contoh penggunaan fungsi translasi
translated_img = translate(gambar_hasil, 10, 20)
plt.imshow(translated_img, cmap='jet')
plt.title('Gambar setelah translasi')
plt.show()

# Contoh penggunaan fungsi refleksi
reflected_img_x = reflect(gambar_hasil, 'x')
plt.imshow(reflected_img_x, cmap='jet')
plt.title('Gambar setelah refleksi sumbu x')
plt.show()

# Contoh penggunaan fungsi rotasi
rotated_img = rotate(gambar_hasil, 45)
plt.imshow(rotated_img, cmap='jet')
plt.title('Gambar setelah rotasi 45 derajat')
plt.show()

# Contoh penggunaan fungsi skala
scaled_img = scale(gambar_hasil, 1.5)
plt.imshow(scaled_img, cmap='jet')
plt.title('Gambar setelah penskalaan')
plt.show()