import os
import pickle
import datetime
import importlib.util

# 2_perceptron_classifier.py 임포트해옴
spec = importlib.util.spec_from_file_location("perceptron", "2_perceptron_classifier.py")
perceptron = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perceptron)
SimpleSLPClassifier = perceptron.SimpleSLPClassifier

def write_log(message, log_file):
    # 화면에 프린트도 하고 텍스트 파일에도 쓰는 함수
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_dataset(dataset_dir):
    # 폴더 뒤져서 이미지 다 읽어오는 부분
    X, y = [], []
    categories = ['game', 'person', 'finance']
    classifier = SimpleSLPClassifier()
    
    for category in categories:
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir): continue
            
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        for filename in files:
            file_path = os.path.join(cat_dir, filename)
            try:
                # 여기서 특징 추출함 (시간 좀 걸림)
                feature = classifier.extract_features(file_path)
                X.append(feature)
                y.append(category)
            except Exception:
                pass
    return X, y

def train_test_split(X, y, test_ratio=0.2):
    import random
    combined = list(zip(X, y))
    random.seed(42) # 결과 맨날 바뀌면 헷갈리니까 고정시킴
    random.shuffle(combined)
    split_index = int(len(combined) * (1 - test_ratio))
    
    # 80퍼센트는 학습용, 20퍼센트는 나중에 테스트하려고 빼둠
    train_data = combined[:split_index]
    test_data = combined[split_index:]
    
    X_train = [item[0] for item in train_data]
    y_train = [item[1] for item in train_data]
    X_test = [item[0] for item in test_data]
    y_test = [item[1] for item in test_data]
    
    return X_train, X_test, y_train, y_test

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset_augmented')
    log_file = os.path.join(base_dir, 'train_log.txt')
    
    # 예전 로그 삭제
    if os.path.exists(log_file):
        os.remove(log_file)
        
    write_log(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 퍼셉트론 학습 스크립트 시작", log_file)
    
    if not os.path.exists(dataset_dir):
        write_log("오류: dataset_augmented 폴더가 없음. 증강부터 먼저 해야됨.", log_file)
        return
        
    write_log(f"읽어올 폴더: {dataset_dir}", log_file)
    write_log("특징 뽑아내는중... (노트북 성능에 따라 시간 좀 걸릴수있음)", log_file)
    
    X, y = load_dataset(dataset_dir)
    write_log(f"특징 추출 끝! 총 {len(X)}개 로드됨.", log_file)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    write_log(f"데이터 쪼개기 끝: 학습용 {len(X_train)}장, 테스트용 {len(X_test)}장", log_file)
    
    # 단층 퍼셉트론 학습 진행해 보기 (로그 확인용)
    model = SimpleSLPClassifier()
    model.fit(X_train, y_train)
    
    # 뽑아놓은 데이터를 pickle로 저장해놔야 다른 스크립트에서 바로 씀
    model_path = os.path.join(base_dir, 'perceptron_model.pkl')
    test_data_path = os.path.join(base_dir, 'test_dataset.pkl')
    
    # 호환성/속도 유지를 위해 raw features를 모델 pkl로 저장 (GUI, Flask가 로딩 시 퍼셉트론 피팅 수행)
    with open(model_path, 'wb') as f:
        pickle.dump((X_train, y_train), f)
        
    with open(test_data_path, 'wb') as f:
        pickle.dump((X_test, y_test), f)
        
    write_log(f"모델용 데이터셋 저장 완료 (퍼셉트론 입력용): {model_path}", log_file)
    write_log(f"테스트용 데이터도 저장함: {test_data_path}", log_file)
    write_log(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 학습 스크립트 다 돌았음", log_file)

if __name__ == '__main__':
    main()
