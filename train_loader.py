import os
import importlib.util

spec = importlib.util.spec_from_file_location("knn", "2_knn_classifier.py")
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)
SimpleKNNClassifier = knn.SimpleKNNClassifier

def load_full_dataset(dataset_dir):
    # 웹서버 켤때마다 매번 데이터 뽑기 귀찮아서 만든 헬퍼 함수
    X, y = [], []
    categories = ['game', 'person', 'finance']
    classifier = SimpleKNNClassifier() 
    
    for category in categories:
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir): continue
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        for filename in files:
            file_path = os.path.join(cat_dir, filename)
            try:
                feature = classifier.extract_features(file_path)
                X.append(feature)
                y.append(category)
            except Exception: pass
    return X, y
