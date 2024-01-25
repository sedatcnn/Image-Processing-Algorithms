import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import os

def esitleme_fonksiyonu(image_path, user_c):
    # Resmi gri tonlamalı olarak oku
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Adaptif eşikleme uygula
    block_size = 11
    adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, user_c)
    
    return adaptive_threshold

def esitleme_otsu_fonksiyonu(image_path):
    # Resmi gri tonlamalı olarak oku
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Otsu eşikleme uygula
    _, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return otsu_threshold

def kerenel_flitered_fonksiyonu(image_path):
    # Resmi oku
    image = cv2.imread(image_path)
    
    # Kenar vurgulama filtresi tanımla
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    # Filtreyi uygula
    sonuc = cv2.filter2D(image, -1, kernel)
    
    return sonuc

def farkli_sinir_fonksiyonu(image_path, border_width):
    # Resmi oku
    image = cv2.imread(image_path)
    
    # Kenar rengi tanımla
    border_color = (0, 0, 0)

    # Resim okunamazsa hata mesajı ver
    if image is None:
        print("Hata: Resim okunamadı.")
        return None

    # Resim boyutlarını al
    height, width, channels = image.shape
    
    # Kenar genişliğini sınırla
    border_width = min(border_width, height // 2, width // 2)

    # Kenarları oluştur
    top_border = np.full((border_width, width, channels), border_color, dtype=np.uint8)
    bottom_border = np.full((border_width, width, channels), border_color, dtype=np.uint8)
    left_border = np.full((height, border_width, channels), border_color, dtype=np.uint8)
    right_border = np.full((height, border_width, channels), border_color, dtype=np.uint8)

    # Resmin üst kısmına üst kenarı ekle
    image[:border_width, :, :] = top_border
    # Resmin alt kısmına alt kenarı ekle
    image[-border_width:, :, :] = bottom_border
    # Resmin sol tarafına sol kenarı ekle
    image[:, :border_width, :] = left_border
    # Resmin sağ tarafına sağ kenarı ekle
    image[:, -border_width:, :] = right_border

    # Sonuç resmi kaydetme yolu
    result_image_path = "sonuc_resmin_kayit_yolu.png"  # Bu kısmı kendi kaydetme mantığınıza uygun şekilde değiştirin
    cv2.imwrite(result_image_path, image)

    return result_image_path

def gamma_fonksiyonu(image_path, gamma_value):
    def apply_gamma_correction(image, gamma):
        # Görüntüyü 0-1 arasında normalize et
        image_normalized = image / 255.0
        
        # Gamma düzeltmesi uygula
        gamma_corrected = np.power(image_normalized, gamma)
        
        # 0-255 arasına geri dönüştür
        gamma_corrected = np.uint8(gamma_corrected * 255)
        
        return gamma_corrected

    # Resmi oku
    image = cv2.imread(image_path)
    
    # Gamma düzeltmesini uygula
    gamma_corrected_image = apply_gamma_correction(image, gamma=gamma_value)
    
    return gamma_corrected_image

def blur(image_path, kernel_size):
    # Resmi oku
    image = cv2.imread(image_path)
    
    # Bulanıklaştırma filtresini uygula
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))
    
    return blurred_image

def median_blur(image_path, kernel_size):
    # Resmi oku
    image = cv2.imread(image_path)
    
    # Kernel boyutunu tek yap, çift sayı olmamalı
    kernel_size = max(1, kernel_size // 2) * 2 + 1
    
    # Medyan bulanıklaştırma uygula
    median_blurred_image = cv2.medianBlur(image, kernel_size)
    
    return median_blurred_image

def box_filter(image_path, kernel_size):
    # Resmi oku
    image = cv2.imread(image_path)
    
    # Kutu filtresini uygula
    box_filtered_image = cv2.boxFilter(image, -1, (kernel_size, kernel_size))
    
    return box_filtered_image

def bilateral_filter(image_path):
    # Resmi oku
    image = cv2.imread(image_path)
    
    # Bilateral filtre uygula
    bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    
    return bilateral_filtered_image

def gaussian_blur(image_path, kernel_size):
    # Resmi oku
    image = cv2.imread(image_path)
    
    # Kernel boyutunu tek yap, çift sayı olmamalı
    kernel_size = max(1, kernel_size // 2) * 2 + 1
    
    # Gaussian bulanıklaştırma uygula
    gaussian_blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return gaussian_blurred_image

def goruntu_keskinlestirme(image_path):
    # Resmi oku
    image = cv2.imread(image_path)

    # Kenar vurgulama filtresi tanımla
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Filtreyi uygula
    result = cv2.filter2D(image, -1, kernel)

    # Orijinal ve filtrelenmiş görüntüleri döndür
    return result

def histogram_fonksiyonu(image_path):
    # Resmi gri tonlamalı olarak oku
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Histogram eşitlemeyi uygula
    equalized_image = cv2.equalizeHist(image)

    # Matplotlib ile iki alt grafik oluştur
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.show()

    return equalized_image

def histogram_esitleme_fonksiyonu(image_path):
    # Resmi oku
    image = cv2.imread(image_path)
    
    # Gri tonlamalı hale getir
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogramı hesapla
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    
    # Gri tonlamalı resmi göster
    cv2.imshow("Gri Tonlamalı Görüntü", image_gray)
    
    # Histogramı göster
    plt.plot(hist)
    plt.title("Histogram Eğrisi")
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Piksel Sayısı")
    plt.show()

    # Bekle ve pencereyi kapat
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gri tonlamalı resmi ve histogram eğrisini döndür
    return image_gray, hist

def sobel_edges(image_path, ksize=3):
    # Gri tonlamalı olarak resmi oku
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Sobel operatörlerini uygula
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Negatif değerleri pozitif yap
    sobelx = np.abs(sobelx)
    sobely = np.abs(sobely)

    # Kenarları birleştir
    edges = cv2.bitwise_or(sobelx, sobely)

    # NaN değerleri sıfıra dönüştür
    edges = np.nan_to_num(edges)

    # uint8 formatına çevir
    edges = np.uint8(edges)

    return edges

def laplacian(image_path):
    # Gri tonlamalı resmi oku
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Laplacian operatörlerini uygula
    laplacian_result1 = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_result1 = np.uint8(np.absolute(laplacian_result1))

    # Gaussian bulanıklaştırma uygula
    img_blurred = cv2.GaussianBlur(image, (3, 3), 0)
    laplacian_result2 = cv2.Laplacian(img_blurred, ddepth=-1, ksize=3)

    # Resimleri ve sonuçları göster
    plt.figure(figsize=(17, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.imshow(laplacian_result1, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.imshow(laplacian_result2, cmap="gray")

    plt.show()

    # Gri tonlamalı resmi ve iki farklı Laplacian sonucunu döndür
    return image, laplacian_result1, laplacian_result2

def canny_edge_detection(image_path, low_threshold=50, high_threshold=150, L2gradient=True):
    # Gri tonlamalı resmi oku
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Canny kenar tespitini uygula
    edges = cv2.Canny(image, low_threshold, high_threshold, L2gradient=L2gradient)

    return edges

def deriche_edge_detection(image_path, alpha=0.5, kernel_size=3):
    # Gri tonlamalı resmi oku
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Deriche çekirdeklerini al
    kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
    deriche_kernel_x = alpha * kx
    deriche_kernel_y = alpha * ky

    # Deriche filtresini uygula
    deriche_x = cv2.filter2D(image, cv2.CV_64F, deriche_kernel_x)
    deriche_y = cv2.filter2D(image, cv2.CV_64F, deriche_kernel_y)

    # Gradient büyüklüğünü kullanarak kenarları hesapla
    edges = np.sqrt(deriche_x**2 + deriche_y**2)

    # Rengi dönüştürmeden önce resmi uint8'e çevir
    edges = np.uint8(edges)

    # Renk dönüştürme
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return edges_rgb

def harris_corner_detection(image_path, corner_quality=0.04, min_distance=10, block_size=3):
    # Resmi oku
    img = cv2.imread(image_path)

    # Resmi gri tonlamalı hale getir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris köşe tespitini uygula
    corners = cv2.cornerHarris(gray, block_size, 3, corner_quality)

    # Köşeleri daha görünür yapmak için genişlet
    corners = cv2.dilate(corners, None)

    # Köşeleri orijinal resimde işaretle
    img[corners > 0.01 * corners.max()] = [0, 0, 255]

    return img

def detect_faces(image_path):
    # Yüz tanıma için önceden eğitilmiş Haar kaskadını yükle
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Resmi oku
    img = cv2.imread(image_path, 0)

    # Varsayılan parametrelerle yüz tespiti yap
    faces1 = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    img_with_faces1 = img.copy()  # Dikdörtgenleri çizmek için kopya oluştur
    for (x, y, w, h) in faces1:
        cv2.rectangle(img_with_faces1, (x, y), (x+w, y+h), (255, 0, 0), 10)

    # Özel parametrelerle yüz tespiti yap
    faces2 = faceCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=6)
    img_with_faces2 = img.copy()  # Dikdörtgenleri çizmek için kopya oluştur
    for (x, y, w, h) in faces2:
        cv2.rectangle(img_with_faces2, (x, y), (x+w, y+h), (255, 0, 0), 10)

    # Alt çizimler oluştur
    f, eksen = plt.subplots(1, 2, figsize=(20, 10))

    # Orijinal görüntüyü, varsayılan parametrelerle dikdörtgenlerle çevrili olarak göster
    eksen[0].imshow(img_with_faces1, cmap="gray")

    # Orijinal görüntüyü, özel parametrelerle dikdörtgenlerle çevrili olarak göster
    eksen[1].imshow(img_with_faces2, cmap="gray")

    # Çizimleri göster
    plt.show()

    return faces1, faces2

def find_and_draw_contours(image_path, canny_threshold1=50, canny_threshold2=150, contour_color=(0, 255, 0), contour_thickness=2):
    # Resmi oku
    img = cv2.imread(image_path)

    # Resmi griye dönüştür
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Canny kenar tespiti uygula
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Kenarlardaki konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Orijinal görüntü üzerine konturları çiz
    cv2.drawContours(img, contours, -1, contour_color, contour_thickness)

    return img

def watershed_segmentation(image_path):
    # Görüntüyü oku
    imgOrj = cv2.imread(image_path)
    
    # Detayları azaltmak için median blur uygula
    imgBlr = cv2.medianBlur(imgOrj, 31)
    
    # Bulanıklaştırılmış görüntüyü gri tonlamalı hale getir
    imgGray = cv2.cvtColor(imgBlr, cv2.COLOR_BGR2GRAY)
    
    # OTSU yöntemi ile eşikleme uygula
    ret, imgTH = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Gürültüyü kaldırmak için morfolojik açma uygula
    kernel = np.ones((5, 5), np.uint8)
    imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)

    # Net arka plan elde etmek için genişletme uygula
    sureBG = cv2.dilate(imgOPN, kernel, iterations=3)

    # Kesin ön planı bulmak için mesafe dönüşümü uygula
    dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)
    ret, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Kesin ön planı uint8'e çevir
    sureFG = np.uint8(sureFG)
    
    # Bilinmeyen bölgeleri tanımla
    unknown = cv2.subtract(sureBG, sureFG)

    # Bağlantılı bileşen etiketleme
    ret, markers = cv2.connectedComponents(sureFG, labels=5)

    # Bilinmeyen bölgeleri etiketle
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed algoritmasını uygula
    markers = cv2.watershed(imgOrj, markers)

    # Contourları bul ve orijinal görüntü üzerine çiz
    contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    imgCopy = imgOrj.copy()
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(imgCopy, contours, i, (255, 0, 0), 5)

    # Matplotlib kullanarak görüntüleri göster
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    axes[0, 0].imshow(cv2.cvtColor(imgOrj, cv2.COLOR_BGR2RGB))

    axes[0, 1].imshow(cv2.cvtColor(imgBlr, cv2.COLOR_BGR2RGB))

    axes[0, 2].imshow(imgGray, cmap='gray')

    axes[1, 0].imshow(imgTH, cmap='gray')

    axes[1, 1].imshow(imgOPN, cmap='gray')

    axes[1, 2].imshow(sureBG, cmap='gray')

    axes[2, 0].imshow(dist_transform, cmap='jet')

    axes[2, 1].imshow(sureFG, cmap='gray')

    axes[2, 2].imshow(cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB))

    plt.show()

    return imgCopy





