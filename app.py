import os
from flask import Flask, render_template, request, jsonify
import importlib.util

# 내 knn 모델 끌어오기
spec = importlib.util.spec_from_file_location("knn", "2_knn_classifier.py")
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)
SimpleKNNClassifier = knn.SimpleKNNClassifier

app = Flask(__name__)
# 웹에서 파일 올리면 잠깐 저장해둘 폴더
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 서버 켤 때 모델 미리 세팅해놓음 (안그러면 렉걸림)
print("서버 켜는중... 데이터 특징 뽑는중이라 좀 걸릴수있음ㅇㅇ")
from train_loader import load_full_dataset
import random

X, y = load_full_dataset('dataset_augmented')
combined = list(zip(X, y))
random.seed(42)  # 새로고침할때마다 정확도 바뀌면 쪽팔리니까 고정
random.shuffle(combined)
split_index = int(len(combined) * 0.8)

train_data, test_data = combined[:split_index], combined[split_index:]
X_train, y_train = [i[0] for i in train_data], [i[1] for i in train_data]
X_test, y_test = [i[0] for i in test_data], [i[1] for i in test_data]

model = SimpleKNNClassifier(k=5)
model.fit(X_train, y_train)

# 발표할때 화면에 띄워줄 용도로 자체 평가 한번 돌림
predictions = model.predict(X_test)
correct = sum(1 for t, p in zip(y_test, predictions) if t == p)
global_accuracy = (correct / len(y_test)) * 100
total_samples = len(X)
print(f"플라스크 서버 준비완료!! 현재 정확도: {global_accuracy:.2f}%")

@app.route('/')
def index():
    return render_template('index.html', accuracy=round(global_accuracy, 2), total=total_samples)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # 방금 올린 파일 특징 바로 뽑아서 예측 돌리기
            feature = model.extract_features(file_path)
            predictions = model.predict([feature])
            result_category = predictions[0]
            
            return jsonify({
                'category': result_category,
                'image_url': f"/static/uploads/{file.filename}"
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
