import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from datetime import datetime
import os
import numpy as np
from Islemler import *  # Islemler modülünden fonksiyonları içe aktar
from ttkthemes import ThemedStyle,ThemedTk
import tkinter.messagebox as messagebox
from customtkinter import CTkButton

class FaceDetectionApp:
    def set_dark_theme(self):
        # Temayı koyu renkli bir tema olarak ayarla
        style = ThemedStyle(self.root)
        style.set_theme("equilux")  # Diğer temaları da deneyebilirsiniz: "scidgrey", "equilux", vb.

        # Belirli widget'lara tema uygula
        style.configure('Blue.TButton', foreground='white', background='#2196F3')  # Blue color
        style.configure('Yellow.TButton', foreground='black', background='#FFEB3B')  # Yellow color
        style.configure('Green.TButton', foreground='white', background='#4CAF50')  # Green color
        style.configure('TLabel', foreground='white', background='#333333')
        style.configure('TEntry', foreground='white', background='#333333')
        style.configure('TFrame', background='#333333')  # Container'lara temayı uygula

        # Canvas widget'larına temayı doğrudan uygula
        self.processed_photo_canvas.configure(bg='#333333')

        # Ana pencere arka plan rengini güncelle
        self.root.configure(bg='#333333')  # Siyah olarak ayarla
       
        self.live_container.configure(bg='#333333')
        self.photo_container.configure(bg='#333333')
        self.original_photo_container.configure(bg='#333333')
        self.processed_photo_container.configure(bg='#333333')

    def __init__(self, root, window_title="Görüntü İşleme"):
        # Ana pencere ve başlık oluştur
        self.root = root
        self.root.wm_iconbitmap("img/image-processing.ico")
        self.root.title(window_title)
        self.user_c = tk.StringVar()
        self.user_c.set("2")

        self.border_width_var = tk.StringVar()
        self.border_width_var.set("20")

        self.gamma_value_var = tk.StringVar()
        self.gamma_value_var.set("2")

        self.kernel_size_var = tk.StringVar()
        self.kernel_size_var.set("5")

        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)

        # Container'lar oluştur
        self.live_container = tk.Frame(root)
        self.live_container.grid(row=0, column=0, padx=10, pady=10)

        self.photo_container = tk.Frame(root)
        self.photo_container.grid(row=0, column=2, padx=10, pady=10)

        self.original_photo_container = tk.Frame(root)
        self.original_photo_container.grid(row=0, column=1, padx=10, pady=10)

        self.processed_photo_container = tk.Frame(root)
        self.processed_photo_container.grid(row=0, column=3, padx=10, pady=10)

        # Video çerçevesinin boyutlarını al
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Yeniden boyutlandırılmış genişlik ve yükseklik
        self.resized_width = 480
        self.resized_height = 360

        # Canvas widget'ını oluştur
        self.canvas = tk.Canvas(self.live_container, width=self.resized_width, height=self.resized_height)
        self.canvas.grid(row=0, column=0, columnspan=2)

        # İkonları yükle
        self.icon_take_photo = Image.open("img/camera.png")
        self.icon_select_photo = Image.open("img/search.png")
        self.icon_save = Image.open("img/save.png")

        # İkonları yeniden boyutlandır (gerekiyorsa)
        icon_size = (24, 24)
        self.icon_take_photo = self.icon_take_photo.resize(icon_size, Image.LANCZOS)
        self.icon_select_photo = self.icon_select_photo.resize(icon_size, Image.LANCZOS)
        self.icon_save = self.icon_save.resize(icon_size, Image.LANCZOS)

        # İkonları Tkinter PhotoImage formatına dönüştür
        self.icon_take_photo = ImageTk.PhotoImage(self.icon_take_photo)
        self.icon_select_photo = ImageTk.PhotoImage(self.icon_select_photo)
        self.icon_save = ImageTk.PhotoImage(self.icon_save)

        # Fotoğraf Çek butonunu oluştur ve ikonu ekle (blue color)
        self.btn_take_photo = ttk.Button(self.live_container, text="Fotoğraf Çek", command=self.take_photo, image=self.icon_take_photo, compound=tk.LEFT)
        self.btn_take_photo.grid(row=1, column=0, pady=10, padx=(0, 5))
        self.btn_take_photo.configure(style='Blue.TButton')  # Set style

        # Fotoğraf Seç butonunu oluştur ve ikonu ekle (yellow color)
        self.btn_select_photo = ttk.Button(self.live_container, text="Fotoğraf Seç", command=self.select_photo, image=self.icon_select_photo, compound=tk.LEFT)
        self.btn_select_photo.grid(row=1, column=1, pady=10, padx=(5, 0))
        self.btn_select_photo.configure(style='Yellow.TButton')  # Set style

        # Kaydet butonunu oluştur ve ikonu ekle (green color)
        self.btn_kaydet = ttk.Button(self.live_container, text="Kaydet", command=self.kaydet_action, image=self.icon_save, compound=tk.LEFT)
        self.btn_kaydet.grid(row=2, column=1, pady=10, padx=(5, 0))
        self.btn_kaydet.configure(style='Green.TButton')  # Set style

        # Yakalanan fotoğrafın Canvas widget'ını oluştur
        self.captured_photo_canvas = tk.Canvas(self.photo_container, width=self.resized_width, height=self.resized_height)
        self.captured_photo_canvas.grid(row=0, column=0, pady=10)

        # İşlenmiş fotoğrafın Canvas widget'ını oluştur
        self.processed_photo_canvas = tk.Canvas(self.processed_photo_container, width=self.resized_width, height=self.resized_height)
        self.processed_photo_canvas.grid(row=1, column=0, pady=10, padx=(75,0))

        # Fotoğraf işleme seçeneklerini içeren ComboBox'ı oluştur
        self.selected_option = tk.StringVar()
        self.selected_option.set("...")

        self.processing_options = ttk.Combobox(self.live_container, textvariable=self.selected_option, width=25)
        self.processing_options['values'] = ("Orijinal", "Adaptive Threshold", "Otsu Threshold", "Kenarlık Ekle", "Blur", "Median Blur", "Box Filter", "Bilateral Filter", "Gaussian Blur", "Görüntü Keskinleştirme", "Gamma Correction", "Histogram", "Histogram Eşitleme","Sobel Kenar Algoritması","Laplacian","Canny Kenar Algoritması","Harris Köşe Algoritması","Deriche Köşe Algoritması","Yüz Algılama Algoritması","Contours Algoritması","Watershed Algoritması",)
        self.processing_options.grid(row=2, column=0, pady=10, padx=(0, 5))

        # ComboBox'tan seçilen olaya işlevi bağla
        self.processing_options.bind("<<ComboboxSelected>>", self.apply_option)

        # Kullanıcı C değeri için izleme ekle
        self.user_c.trace_add('write', self.update_params)
        self.label_user_c = ttk.Label(self.live_container, text="C Değeri:")
        self.label_user_c.grid(row=3, column=0, pady=10, padx=(0, 5))
        self.entry_user_c = ttk.Entry(self.live_container, textvariable=self.user_c)
        self.entry_user_c.grid(row=3, column=1, pady=10, padx=(5, 0))

        # Kenarlık Kalınlığı değeri için izleme ekle
        self.border_width_var.trace_add('write', self.update_params)
        self.param_kalinlik_label = ttk.Label(self.live_container, text="Kalınlık Değeri:")
        self.param_kalinlik_label.grid(row=4, column=0, pady=10, padx=(0, 5))
        self.param_kalinlik_entry = ttk.Entry(self.live_container, textvariable=self.border_width_var)
        self.param_kalinlik_entry.grid(row=4, column=1, pady=10, padx=(5, 0))

        # Gamma Değeri için izleme ekle
        self.gamma_value_var.trace_add('write', self.update_params)
        self.param_gamma_label = ttk.Label(self.live_container, text="Gamma Değeri:")
        self.param_gamma_label.grid(row=5, column=0, pady=10, padx=(0, 5))
        self.param_gamma_entry = ttk.Entry(self.live_container, textvariable=self.gamma_value_var)
        self.param_gamma_entry.grid(row=5, column=1, pady=10, padx=(5, 0))

        # Kernel Değeri için izleme ekle
        self.kernel_size_var.trace_add('write', self.update_params)
        self.param_kernel_label = ttk.Label(self.live_container, text="Kernel Değeri:")
        self.param_kernel_label.grid(row=6, column=0, pady=10, padx=(0, 5))
        self.param_kernel_entry = ttk.Entry(self.live_container, textvariable=self.kernel_size_var)
        self.param_kernel_entry.grid(row=6, column=1, pady=10, padx=(5, 0))

        # Orijinal fotoğrafın Canvas widget'ını oluştur
        self.original_photo_canvas = tk.Canvas(self.original_photo_container, width=self.resized_width, height=self.resized_height)
        self.original_photo_canvas.grid(row=0, column=0, pady=10,padx=(5,0))

        # İşlenmemiş fotoğrafın Canvas widget'ını oluştur
        self.processed_photo_canvas = tk.Canvas(self.processed_photo_container,width=self.resized_width, height=self.resized_height)
        self.processed_photo_canvas.grid(row=1, column=0, pady=10,padx=(75,0))

        # C Değeri (User C) için bilgi penceresi
        frame_user_c_info = ttk.Frame(self.live_container)
        frame_user_c_info.grid(row=3, column=2, pady=10, padx=(2, 0))

        self.label_user_c_info = ttk.Label(frame_user_c_info)
        self.label_user_c_info.grid(row=0, column=0)

        question_mark_icon_c = CTkButton(frame_user_c_info, text="?", cursor="question_arrow", width=1, height=1, command=self.show_c_info,)
        question_mark_icon_c.grid(row=0, column=1)

        # Kalınlık Değeri (Border Width) için bilgi penceresi
        frame_kalinlik_info = ttk.Frame(self.live_container)
        frame_kalinlik_info.grid(row=4, column=2, pady=10, padx=(2, 0))

        self.param_kalinlik_label_info = ttk.Label(frame_kalinlik_info)
        self.param_kalinlik_label_info.grid(row=0, column=0)

        question_mark_icon_kalinlik = CTkButton(frame_kalinlik_info, text="?", cursor="question_arrow", width=1, height=1, command=self.show_kalinlik_info)
        question_mark_icon_kalinlik.grid(row=0, column=1)

        # Gamma Değeri (Gamma Value) için bilgi penceresi
        frame_gamma_info = ttk.Frame(self.live_container)
        frame_gamma_info.grid(row=5, column=2, pady=10, padx=(2, 0))

        self.param_gamma_label_info = ttk.Label(frame_gamma_info)
        self.param_gamma_label_info.grid(row=0, column=0)

        question_mark_icon_gamma = CTkButton(frame_gamma_info, text="?", cursor="question_arrow", width=1, height=1, command=self.show_gamma_info)
        question_mark_icon_gamma.grid(row=0, column=1)

        # Kernel Değeri (Kernel Size) için bilgi penceresi
        frame_kernel_info = ttk.Frame(self.live_container)
        frame_kernel_info.grid(row=6, column=2, pady=10, padx=(2, 0))

        self.param_kernel_label_info = ttk.Label(frame_kernel_info)
        self.param_kernel_label_info.grid(row=0, column=0)

        question_mark_icon_kernel = CTkButton(frame_kernel_info, text="?", cursor="question_arrow", width=1, height=1, command=self.show_kernel_info)
        question_mark_icon_kernel.grid(row=0, column=1)

        # Güncelleme fonksiyonunu belirli bir süre boyunca çağır
        self.delay = 10
        self.update()

        # Pencere kapatıldığında çalışacak fonksiyonu belirle
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # Temayı koyu renkli tema olarak ayarla
        self.set_dark_theme()

        self.captured_photo_path = None
        
    def show_c_info(self):
        # C Değeri için bilgi penceresini göster
        messagebox.showinfo("C Değeri", "Her pikselin eşik değerini ayarlamak için kullanılır!")

    def show_kalinlik_info(self):
        # Kalınlık Değeri için bilgi penceresini göster
        messagebox.showinfo("Kalınlık Değeri", "Kenar genişliği belirlemek için kullanılır!")

    def show_gamma_info(self):
        # Gamma Değeri için bilgi penceresini göster
        messagebox.showinfo("Gamma Değeri", "Gama değerini ayarlamak için kullanılır (1.0 = orijinal, <1.0 = aydınlatma, >1.0 = karartma)!")

    def show_kernel_info(self):
         # Kernel Değeri için bilgi penceresini göster
         messagebox.showinfo("Kernel Değeri", "Bulanıklaştırma işlemleri için kernel boyutunu belirlemede kullanılır!")  
    def update_params(self, *args):
        # Bu fonksiyon, herhangi bir parametre değiştiğinde çağrılacaktır
        if self.selected_option.get() == "Adaptive Threshold":
            # Eğer seçilen seçenek Adaptive Threshold ise, resmi güncelle
            self.apply_option()

        filter_options = ["Blur", "Median Blur", "Box Filter", "Gaussian Blur"]
        if self.selected_option.get() in filter_options:
            self.apply_option()

        # Diğer parametreler için benzer koşullar
        if self.selected_option.get() ==  "Kenarlık Ekle":
            self.apply_option()

        if self.selected_option.get() ==  "Gamma Correction":
            self.apply_option()

                                          
    def apply_option(self, event=None):
        if self.captured_photo_path:
            selected_option = self.selected_option.get()
            
            # Seçilen seçenek Adaptive Threshold ise, C değeri için girişin durumunu güncelle
            self.entry_user_c.config(state='normal' if selected_option == "Adaptive Threshold" else 'disabled')
            # Seçilen seçenek Kenarlık Ekle ise, kalınlık değeri için girişin durumunu güncelle
            self.param_kalinlik_entry.config(state='normal' if selected_option == "Kenarlık Ekle" else 'disabled')
            # Seçilen seçenek Gamma Correction ise, gamma değeri için girişin durumunu güncelle
            self.param_gamma_entry.config(state='normal' if selected_option == "Gamma Correction" else 'disabled')
            # Seçilen seçenek Blur, Median Blur, Box Filter veya Gaussian Blur ise, kernel değeri için girişin durumunu güncelle
            self.param_kernel_entry.config(state='normal' if selected_option in ["Blur", "Median Blur", "Box Filter", "Gaussian Blur"] else 'disabled')

            # Kullanıcıdan alınan parametreleri güncelle
            user_c = float(self.user_c.get()) if self.user_c.get() else 2
            border_width = int(self.border_width_var.get()) if self.border_width_var.get() else 20
            gamma_value = float(self.gamma_value_var.get()) if self.gamma_value_var.get() else 2
            kernel_size = int(self.kernel_size_var.get()) if self.kernel_size_var.get() else 5
            result_image_path = None

            # Seçilen seçeneklere göre işlemleri uygula
            if selected_option == "...":
                result_image_path = esitleme_fonksiyonu(self.captured_photo_path)
            elif selected_option == "Adaptive Threshold":
                user_c = float(self.entry_user_c.get()) if self.entry_user_c.get() else 2
                result_image_path = esitleme_fonksiyonu(self.captured_photo_path, user_c)
            elif selected_option == "Otsu Threshold":
                result_image_path = esitleme_otsu_fonksiyonu(self.captured_photo_path)
            elif selected_option == "Kenarlık Ekle":
                result_image_path = farkli_sinir_fonksiyonu(self.captured_photo_path, border_width)
            elif selected_option in ["Blur", "Median Blur", "Box Filter", "Gaussian Blur"]:
                if selected_option == "Blur":
                    result_image_path = blur(self.captured_photo_path, kernel_size)
                elif selected_option == "Median Blur":
                    result_image_path = median_blur(self.captured_photo_path, kernel_size)
                elif selected_option == "Box Filter":
                    result_image_path = box_filter(self.captured_photo_path, kernel_size)
                elif selected_option == "Gaussian Blur":
                    result_image_path = gaussian_blur(self.captured_photo_path, kernel_size)
            elif selected_option == "Bilateral Filter":
                result_image_path = bilateral_filter(self.captured_photo_path)
            elif selected_option == "Gamma Correction":
                result_image_path = gamma_fonksiyonu(self.captured_photo_path, gamma_value)
            elif selected_option == "Histogram":
                result_image_path = histogram_fonksiyonu(self.captured_photo_path)
            elif selected_option == "Histogram Eşitleme":
                result_image_path = histogram_esitleme_fonksiyonu(self.captured_photo_path)                
            elif selected_option == "Sobel Kenar Algoritması":
                result_image_path = sobel_edges(self.captured_photo_path)
            elif selected_option == "Laplacian":
                result_image_path = laplacian(self.captured_photo_path)    
            elif selected_option == "Canny Kenar Algoritması":
                result_image_path = canny_edge_detection(self.captured_photo_path)
            elif selected_option == "Harris Köşe Algoritması":
                result_image_path = harris_corner_detection(self.captured_photo_path)
            elif selected_option == "Deriche Köşe Algoritması":
                result_image_path = deriche_edge_detection(self.captured_photo_path)
            elif selected_option == "Görüntü Keskinleştirme":
                result_image_path = goruntu_keskinlestirme(self.captured_photo_path)                
            elif selected_option == "Yüz Algılama Algoritması":
                result_image_path = detect_faces(self.captured_photo_path)
            elif selected_option == "Contours Algoritması":
                result_image_path = find_and_draw_contours(self.captured_photo_path)
            elif selected_option == "Watershed Algoritması":
                result_image_path = watershed_segmentation(self.captured_photo_path)

            # Sonuç resmi varsa göster
            if result_image_path is not None:
                if isinstance(result_image_path, str) and os.path.exists(result_image_path):
                    # result_image_path geçerli bir dosya yolu mu diye kontrol et
                    self.display_captured_photo(result_image_path)

                    if selected_option != "...":
                        # Display the original photo only if the option is not "..."
                        original_image = cv2.imread(self.captured_photo_path)
                        self.display_original_photo(original_image)
                elif isinstance(result_image_path, np.ndarray) and result_image_path.size > 0:
                    # Check if result_image_path is a valid NumPy array
                    self.display_captured_photo(result_image_path)
                    self.display_original_photo(cv2.imread(self.captured_photo_path))
        else:
            print("Lütfen önce bir fotoğraf çekin veya seçin.")
        

    def take_photo(self):
                # Kameradan fotoğraf çekme işlemi

        _, frame = self.cap.read()
        frame = cv2.flip(frame, +1)

        save_dir = "DATA"
        os.makedirs(save_dir, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"{save_dir}/foto_{timestamp}.png"
        cv2.imwrite(file_name, frame)
        print(f"Fotoğraf başarıyla kaydedildi: {file_name}")
        # Çekilen fotoğrafı göster

        self.display_captured_photo(file_name)
        self.display_original_photo(frame)

        self.captured_photo_path = file_name

    def select_photo(self):
                # Kullanıcının bir fotoğraf seçmesini sağlama işlemi

        file_path = filedialog.askopenfilename(title="Fotoğraf Seçin", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
                        # Seçilen fotoğrafı göster

            self.display_captured_photo(file_path)
            self.display_original_photo(cv2.imread(file_path))

            self.captured_photo_path = file_path

    def kaydet_action(self):
        if self.captured_photo_path:
            selected_option = self.selected_option.get()
            result_image = None

            if selected_option == "...":
                result_image = esitleme_fonksiyonu(self.captured_photo_path)
            elif selected_option == "Adaptive Threshold":
                user_c = float(self.user_c.get()) if self.user_c.get() else 2
                result_image = esitleme_fonksiyonu(self.captured_photo_path, user_c)
            elif selected_option == "Otsu Threshold":
                result_image = esitleme_otsu_fonksiyonu(self.captured_photo_path)
            elif selected_option == "Kenarlık Ekle":
                border_width = int(self.border_width_var.get()) if self.border_width_var.get() else 20
                result_image = farkli_sinir_fonksiyonu(self.captured_photo_path, border_width)
            elif selected_option == "Blur":
                kernel_size = int(self.kernel_size_var.get()) if self.kernel_size_var.get() else 5
                result_image = blur(self.captured_photo_path, kernel_size)
            elif selected_option == "Median Blur":
                kernel_size = int(self.kernel_size_var.get()) if self.kernel_size_var.get() else 5
                result_image = median_blur(self.captured_photo_path, kernel_size)
            elif selected_option == "Box Filter":
                kernel_size = int(self.kernel_size_var.get()) if self.kernel_size_var.get() else 5
                result_image = box_filter(self.captured_photo_path, kernel_size)
            elif selected_option == "Gaussian Blur":
                kernel_size = int(self.kernel_size_var.get()) if self.kernel_size_var.get() else 5
                result_image = gaussian_blur(self.captured_photo_path, kernel_size)
            elif selected_option == "Bilateral Filter":
                result_image = bilateral_filter(self.captured_photo_path)
            elif selected_option == "Gamma Correction":
                gamma_value = float(self.gamma_value_var.get()) if self.gamma_value_var.get() else 2
                result_image = gamma_fonksiyonu(self.captured_photo_path, gamma_value)
            elif selected_option == "Histogram":
                result_image = histogram_fonksiyonu(self.captured_photo_path)
            elif selected_option == "Histogram Eşitleme":
                result_image = histogram_esitleme_fonksiyonu(self.captured_photo_path)
            elif selected_option == "Sobel Kenar Algoritması":
                result_image = sobel_edges(self.captured_photo_path)
            elif selected_option == "Laplacian":
                result_image = laplacian(self.captured_photo_path)
            elif selected_option == "Canny Kenar Algoritması":
                result_image = canny_edge_detection(self.captured_photo_path)
            elif selected_option == "Harris Köşe Algoritması":
                result_image = harris_corner_detection(self.captured_photo_path)
            elif selected_option == "Deriche Köşe Algoritması":
                result_image = deriche_edge_detection(self.captured_photo_path)
            elif selected_option == "Görüntü Keskinleştirme":
                result_image = goruntu_keskinlestirme(self.captured_photo_path)
            elif selected_option == "Yüz Algılama Algoritması":
                result_image = detect_faces(self.captured_photo_path)
            elif selected_option == "Contours Algoritması":
                result_image = find_and_draw_contours(self.captured_photo_path)
            elif selected_option == "Watershed Algoritması":
                result_image = watershed_segmentation(self.captured_photo_path)
            if result_image is not None:
                save_dir = "Processed_Images"
                os.makedirs(save_dir, exist_ok=True)
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                processed_file_name = f"{save_dir}/processed_foto_{timestamp}.png"

                if isinstance(result_image, str):
                    cv2.imwrite(processed_file_name, cv2.imread(result_image))
                elif isinstance(result_image, np.ndarray):
                    cv2.imwrite(processed_file_name, result_image)

            full_path = os.path.abspath(processed_file_name)
            print(f"Processed fotoğraf başarıyla kaydedildi: {full_path}")

            # Display a message box with the saved file path
            messagebox.showinfo("Success", f"Processed fotoğraf başarıyla kaydedildi:\n{full_path}")

        else:
            messagebox.showwarning("Warning", "Lütfen önce bir fotoğraf çekin veya seçin.")




    def display_captured_photo(self, image):
                # İşlenmiş fotoğrafı kaydetme işlemi

        if isinstance(image, str):
            image = cv2.imread(image)
        elif not isinstance(image, (np.ndarray, np.generic)):
            print("Geçersiz resim formatı.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(self.resized_height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, self.resized_height))

        photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))

        self.captured_photo_canvas.config(width=new_width, height=self.resized_height)
        self.captured_photo_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.captured_photo_canvas.photo = photo

    def display_original_photo(self, image):
                # İşlenmiş fotoğrafı kaydetme işlemi

        if not isinstance(image, (np.ndarray, np.generic)):
            print("Geçersiz resim formatı.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(self.resized_height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, self.resized_height))

        photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))

        self.original_photo_canvas.config(width=new_width, height=self.resized_height)
        self.original_photo_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.original_photo_canvas.photo = photo

    def update(self):
                # İşlenmiş fotoğrafı kaydetme işlemi

        _, frame = self.cap.read()
        frame = cv2.flip(frame, +1)

        if frame is not None:
            frame = cv2.resize(frame, (self.resized_width, self.resized_height))
            self.photo = self.convert_to_tkimage(frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.after(self.delay, self.update)

    def convert_to_tkimage(self, frame):
        # OpenCV formatındaki bir görüntüyü Tkinter formatına dönüştürme işlemi
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(image=img)
        return photo

    def on_close(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = ThemedTk(theme="equilux")  # Use the ThemedTk for better high-DPI scaling
    app = FaceDetectionApp(root, window_title="Face Detection App")
    app.set_dark_theme()  
    root.mainloop()
