import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    import tkinter as tk
    from tkinter import filedialog
    from PIL import Image, ImageTk

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# إعدادات المعالجة
IMG_SIZE = 64


def preprocess_image(img_path_or_img):
    """تحضير الصورة: تغيير الحجم وتحويلها للتدرج الرمادي"""
    if isinstance(img_path_or_img, str):
        img = cv2.imread(img_path_or_img)
    else:
        img = img_path_or_img

    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # تحسين التباين يساعد في تقليل الانحياز
        img = cv2.equalizeHist(img)
        return img.flatten()
    return None


def load_dataset(lion_folder, tiger_folder):
    """تحميل البيانات مع مضاعفتها نستخدمها قبل التدريب لتحسين الاداء (Augmentation) لتحسين الدقة"""
    images = []
    labels = []

    folders = [(lion_folder, 0), (tiger_folder, 1)]

    for folder, label in folders:
        for filename in os.listdir(folder):#اسماء الملفات
            path = os.path.join(folder, filename)#ربط وتجمبع اجزاء مسار الملف
            img = cv2.imread(path)
            if img is not None:
                # إضافة الصورة الأصلية
                flat = preprocess_image(img)#يحول الصوزه الئ الاقام
                images.append(flat)
                labels.append(label)

                # إضافة نسخة مقلوبة (لتحسين التوازن ومنع الانحياز لفئة واحدة)
                flipped = cv2.flip(img, 1)#1 mean قلب افقي يمين
                images.append(preprocess_image(flipped))
                labels.append(label)

    return np.array(images), np.array(labels)


class IntegratedApp:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("مشروع تصنيف الأسود والنم"
                        "ر - Naive Bayes")
        self.root.geometry("550x650")
        self.root.configure(bg="#F4F4F9")

        # واجهة المستخدم
        tk.Label(self.root, text="واجهة اختبار النموذج التفاعلية", font=("Arial", 18, "bold"), bg="#F4F4F9" ,
                 fg="#2B2D42").pack(pady=20)

        self.btn_browse = tk.Button(self.root, text="اختر صورة للاختبار", command=self.browse_image,
                                    font=("Arial", 12, "bold"), bg="#EF233C", fg="white", padx=20, pady=10)
        self.btn_browse.pack(pady=10)

        self.canvas = tk.Label(self.root, bg="white", borderwidth=2, relief="solid")
        self.canvas.pack(pady=20)

        self.label_result = tk.Label(self.root, text="النتيجة ستظهر هنا", font=("Arial", 16, "bold"), bg="#F4F4F9",
                                     fg="#4A4E69")
        self.label_result.pack(pady=20)

    def browse_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img_pil = Image.open(file_path)
            img_pil = img_pil.resize((250, 250))
            img_tk = ImageTk.PhotoImage(img_pil)
            self.canvas.configure(image=img_tk)
            self.canvas.image = img_tk

            flat_img = preprocess_image(file_path)
            if flat_img is not None:
                prediction = self.model.predict([flat_img])[0]
                result = "أسد (Lion)" if prediction == 0 else "نمر (Tiger)"
                color = "#2B2D42" if prediction == 0 else "#EF233C"
                self.label_result.config(text=f"التصنيف المتوقع: {result}", fg=color)

    def run(self):
        self.root.mainloop()


def main():
    lion_folder = 'dataset/lions'
    tiger_folder = 'dataset/tigers'

    if not os.path.exists(lion_folder) or not os.path.exists(tiger_folder):
        print("خطأ: تأكد من وجود مجلد dataset وبداخله lions و tigers.")
        return

    print("--- 1. مرحلة تحميل البيانات وتدريب النموذج ---")
    X, y = load_dataset(lion_folder, tiger_folder)

    # تقسيم البيانات للتقييم (مثل الكود الأول)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    print("\n--- 2. مخرجات تقييم الأداء (للمناقشة) ---")
    y_pred = model.predict(X_test)
    print(f"دقة النموذج: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nتقرير التصنيف التفصيلي:")
    print(classification_report(y_test, y_pred, target_names=['Lions', 'Tigers']))
    print("\nمصفوفة الارتباك (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    # إعادة التدريب على كامل البيانات لضمان أفضل أداء في الواجهة
    model.fit(X, y)

    if GUI_AVAILABLE:
        print("\n--- 3. تشغيل واجهة الاختبار التفاعلية ---")
        app = IntegratedApp(model)
        app.run()
    else:
        print("\n[تنبيه] مكتبة الواجهة غير متوفرة، تم عرض النتائج النصية فقط.")


if __name__ == "__main__":
    main()
