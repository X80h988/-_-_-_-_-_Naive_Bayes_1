# Lions vs Tigers Classification | تصنيف الأسود والنمور

مشروع تعلم آلة (Machine Learning) يهدف إلى تصنيف الصور إلى فئتين: "أسود" و "نمور"، باستخدام خوارزمية Naive Bayes الاحتمالية.

## 📌 نظرة عامة
تم تطوير هذا المشروع لتطبيق خوارزمية Naive Bayes (بايز الساذجة) على بيانات الصور. تعتمد الخوارزمية على حساب الاحتمالات الإحصائية لكل ميزة (Pixel/Feature) لتصنيف الصورة الجديدة إلى الفئة الأكثر احتمالاً. المشروع مخصص للأغراض التعليمية والبحثية لفهم أساسيات تصنيف الصور باستخدام الخوارزميات الاحتمالية.

## 📊 البيانات (Dataset)
نظراً للحجم الكبير لملفات البيانات، لم يتم رفع الداتاسيت إلى هذا المستودع. يحتوي هذا المستودع على الكود المصدري لتدريب المودل وتقييمه.
* **وصف البيانات:** صور ملونة/رمادية للأسود والنمور، تمت معالجتها وتحويلها إلى مصفوفات رقمية (Feature Extraction) لتلائم خوارزمية Naive Bayes.

## 🧠 التقنيات وخوارزمية المودل (Tech Stack)
- اللغة: Python
- المكتبات: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, OpenCV/PIL
- الخوارزمية: Naive Bayes (GaussianNB / MultinomialNB)
- نوع المشكلة: تصنيف ثنائي (Binary Classification)
- المقاييس المستخدمة: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 🚀 طريقة التشغيل
1. قم بتثبيت المكتبات المطلوبة:
   ```bash
   pip install -r requirements.txt
قم بتشغيل ملف الـ Jupyter Notebook لتشغيل الكود خطوة بخطوة:
jupyter notebook lions_tigers_naive_bayes.ipynb
⚠️ إخلاء مسؤولية (Disclaimer)
هذا المشروع تم تطويره لأغراض تعليمية وبحثية فقط، ويهدف إلى استعراض قدرات خوارزمية Naive Bayes في تصنيف الصور.
👨‍💻 المطور
· GitHub: @X80h988

***

machine-learning, python, naive-bayes, image-classification, data-science, computer-vision
