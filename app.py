import os
import shutil
import base64
import io
import zipfile
import json
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import cv2

app = Flask(__name__)

# تنظیمات پوشه‌ها
UPLOAD_FOLDER = 'uploads'
INDEX_FOLDER = 'index'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['INDEX_FOLDER'] = INDEX_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 مگابایت

# ذخیره وضعیت در حافظه (برای یک کاربر)
app_state = {
    'clusters': [],
    'face_data': []
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_folders():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(INDEX_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    ensure_folders()
    
    # پاک کردن آپلودهای قبلی
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))
    
    files = request.files.getlist('images')
    saved_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            saved_files.append(filepath)
    
    if not saved_files:
        return jsonify({'error': 'هیچ فایل معتبری آپلود نشد'}), 400
    
    # استخراج چهره‌ها و embedding ها
    embeddings = []
    face_info = []
    
    for img_path in saved_files:
        try:
            # تشخیص چهره و استخراج ویژگی
            result = DeepFace.represent(
                img_path=img_path,
                model_name='Facenet512',
                detector_backend='retinaface',
                enforce_detection=False,
                align=True
            )
            
            # خواندن تصویر برای برش چهره
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for face_data in result:
                embedding = face_data['embedding']
                facial_area = face_data['facial_area']
                
                # برش ناحیه چهره
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                face_crop = img_rgb[max(0, y):min(img_rgb.shape[0], y+h), 
                                   max(0, x):min(img_rgb.shape[1], x+w)]
                
                if face_crop.size == 0:
                    continue
                
                embeddings.append(embedding)
                face_info.append({
                    'source_image': img_path,
                    'face_crop': face_crop,
                    'filename': os.path.basename(img_path)
                })
                
        except Exception as e:
            print(f"خطا در پردازش {img_path}: {e}")
            continue
    
    if len(embeddings) < 2:
        return jsonify({'error': 'تعداد چهره کافی برای خوشه‌بندی یافت نشد'}), 400
    
    # خوشه‌بندی با DBSCAN
    embeddings_array = np.array(embeddings)
    
    # نرمال‌سازی برای فاصله کسینوسی
    embeddings_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    clustering = DBSCAN(eps=0.45, min_samples=1, metric='cosine').fit(embeddings_norm)
    labels = clustering.labels_
    
    # ساخت خوشه‌ها
    unique_labels = set(labels)
    clusters = []
    
    for label in unique_labels:
        cluster_indices = [i for i, l in enumerate(labels) if l == label]
        cluster_faces = [face_info[i] for i in cluster_indices]
        
        # انتخاب حداکثر ۳ نمونه برای نمایش
        samples = cluster_faces[:3]
        sample_images = []
        
        for face in samples:
            # تبدیل به base64
            pil_img = Image.fromarray(face['face_crop'])
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            sample_images.append(f"data:image/jpeg;base64,{img_str}")
        
        clusters.append({
            'id': int(label),
            'count': len(cluster_faces),
            'samples': sample_images,
            'faces': cluster_faces  # ذخیره برای مرحله بعد
        })
    
    app_state['clusters'] = clusters
    app_state['face_data'] = face_info
    app_state['labels'] = labels.tolist()
    
    # آماده‌سازی پاسخ (بدون داده‌های سنگین)
    response_clusters = []
    for c in clusters:
        response_clusters.append({
            'id': c['id'],
            'count': c['count'],
            'samples': c['samples']
        })
    
    return jsonify({
        'success': True,
        'clusters': response_clusters,
        'total_faces': len(embeddings)
    })

@app.route('/save_names', methods=['POST'])
def save_names():
    data = request.get_json()
    names_mapping = data.get('names', {})  # {cluster_id: name}
    
    if not names_mapping:
        return jsonify({'error': 'نام‌ها ارسال نشده‌اند'}), 400
    
    ensure_folders()
    
    # پاک کردن پوشه index قبلی
    if os.path.exists(INDEX_FOLDER):
        shutil.rmtree(INDEX_FOLDER)
    os.makedirs(INDEX_FOLDER)
    
    labels = app_state.get('labels', [])
    face_info = app_state.get('face_data', [])
    
    # ساخت mapping از index چهره به نام
    face_to_name = {}
    for i, label in enumerate(labels):
        if str(label) in names_mapping:
            face_to_name[i] = names_mapping[str(label)]
    
    # کپی فایل‌ها به پوشه‌های مربوطه
    processed = {}
    for i, face in enumerate(face_info):
        if i not in face_to_name:
            continue
        
        name = face_to_name[i]
        name_folder = os.path.join(INDEX_FOLDER, name)
        os.makedirs(name_folder, exist_ok=True)
        
        source = face['source_image']
        dest = os.path.join(name_folder, face['filename'])
        
        # اگر فایل تکراری است، نام جدید بساز
        counter = 1
        original_dest = dest
        while os.path.exists(dest):
            name_part, ext = os.path.splitext(original_dest)
            dest = f"{name_part}_{counter}{ext}"
            counter += 1
        
        shutil.copy2(source, dest)
        processed[name] = processed.get(name, 0) + 1
    
    return jsonify({
        'success': True,
        'message': f"تصاویر {len(processed)} شخص ذخیره شد",
        'details': processed
    })

@app.route('/download', methods=['GET'])
def download_output():
    if not os.path.exists(INDEX_FOLDER) or not os.listdir(INDEX_FOLDER):
        return jsonify({'error': 'خروجی موجود نیست'}), 400
    
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(INDEX_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, INDEX_FOLDER)
                zf.write(file_path, arcname)
    
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='face_clusters.zip'
    )

@app.route('/clear', methods=['POST'])
def clear_all():
    # پاک کردن پوشه‌ها
    for folder in [UPLOAD_FOLDER, INDEX_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    
    # پاک کردن وضعیت
    app_state['clusters'] = []
    app_state['face_data'] = []
    app_state['labels'] = []
    
    return jsonify({'success': True, 'message': 'همه داده‌ها پاک شدند'})

if __name__ == '__main__':
    ensure_folders()
    print("=" * 50)
    print("سرور در حال اجرا روی http://localhost:5000")
    print("برای توقف Ctrl+C را بزنید")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
