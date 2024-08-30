import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

def buat_profil_restoran(nama_restoran, nama_file):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt = f"""
    Buat profil lengkap untuk sebuah restoran bernama {nama_restoran}, termasuk yang berikut:
    - Nama Restoran
    - Alamat
    - Informasi Kontak
    - Email
    - Menu dengan deskripsi, harga, dan ketersediaan
    - Catatan khusus atau fitur dari restoran tersebut.
    Pastikan formatnya mirip dengan yang digunakan dalam profil perusahaan untuk bisnis, terstruktur dan jelas.
    """
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Anda adalah asisten yang membantu membuat profil restoran yang rinci dalam format terstruktur."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    hasil_respon = response_data['choices'][0]['message']['content'].strip()

    # Simpan hasilnya ke file .txt
    with open(nama_file, 'a') as file:
        file.write(hasil_respon + '\n\n')
    
    return hasil_respon

if __name__ == '__main__':
    nama_restoran = "Warung Nasi Jamblang"
    hasil = buat_profil_restoran(nama_restoran, 'Profil_Restoran.txt')
    print(f'Profil yang dihasilkan untuk {nama_restoran}:')
    print(hasil)
    print("Profil telah disimpan ke Profil_Restoran.txt.")
