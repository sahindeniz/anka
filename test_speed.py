#!/usr/bin/env python3
"""
Astro Maestro Pro Hız Testi
"""

import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("🚀 ASTRO MAESTRO PRO HIZ TESTİ")
print("=" * 50)

# 1. Sistem bilgileri
import multiprocessing
import psutil

num_cores = multiprocessing.cpu_count()
mem = psutil.virtual_memory()

print(f"\n📊 SİSTEM BİLGİLERİ:")
print(f"   CPU Çekirdek: {num_cores}")
print(f"   RAM: {mem.total / (1024**3):.1f} GB")
print(f"   Kullanılabilir RAM: {mem.available / (1024**3):.1f} GB")

# 2. OpenCV optimizasyon durumu
print(f"\n🔧 OPENCV OPTİMİZASYONU:")
print(f"   OpenCV versiyonu: {cv2.__version__}")
print(f"   Optimizasyon aktif: {cv2.useOptimized()}")
cv2.setNumThreads(num_cores - 1)
print(f"   Thread sayısı: {cv2.getNumThreads()}")

# 3. Test görüntüsü oluştur
print(f"\n📸 TEST GÖRÜNTÜSÜ OLUŞTURULUYOR...")
h, w = 2000, 2000
test_img = np.random.rand(h, w, 3).astype(np.float32)
print(f"   Boyut: {w} x {h} = {w*h/1e6:.1f} MP")

# 4. Arka plan modülünü test et
print(f"\n🌌 ARKA PLAN MODÜLÜ TESTİ:")

try:
    from processing.background import remove_gradient_dispatch
    
    # Numba kontrolü
    try:
        from numba import jit
        print(f"   Numba JIT: ✅ Aktif")
    except ImportError:
        print(f"   Numba JIT: ❌ Kurulu değil (pip install numba)")
    
    # Hız testi
    times = []
    for i in range(3):
        start = time.time()
        result = remove_gradient_dispatch(test_img, method='auto', progress_cb=None)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Deneme {i+1}: {elapsed:.2f} saniye")
    
    avg_time = np.mean(times)
    print(f"   ⏱️ Ortalama süre: {avg_time:.2f} saniye")
    print(f"   📐 Sonuç boyutu: {result.shape}")
    
except Exception as e:
    print(f"   ❌ HATA: {e}")
    import traceback
    traceback.print_exc()

# 5. Sonuç
print(f"\n✅ HIZ TESTİ TAMAMLANDI!")
print("=" * 50)