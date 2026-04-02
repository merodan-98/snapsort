import os
import random
from typing import List, Tuple

from importlib.util import spec_from_file_location, module_from_spec

# 2_knn_classifier.py 모듈 불러오기 (같은 폴더)
import importlib.util
spec = importlib.util.spec_from_file_location("knn", "2_knn_classifier.py")
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)
SimpleKNNClassifier = knn.SimpleKNNClassifier

def load_dataset(dataset_dir) -> Tuple[List, List]:
    """
    증강된 데이터셋 폴더에서 모든 이미지를 순회하며 특징 벡터와 라벨을 추출합니다.
    """
    X = []
    y = []
    
    categories = ['game', 'person', 'finance']
    classifier = SimpleKNNClassifier() # 특징 추출 기능만 먼저 빌려씀
    
    print("\n[System] 데이터셋에서 이미지 특징을 추출합니다 (이 작업은 다소 시간이 걸릴 수 있습니다)...")
    
    for category in categories:
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir):
            continue
            
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        # 터미널에 진행률을 예쁘게 표시
        try:
            from tqdm import tqdm
            iterable_files = tqdm(files, desc=f"{category} 특징 추출")
        except ImportError:
            iterable_files = files
            print(f"-> {category} 폴더에서 {len(files)}장 추출 중...")

        for filename in iterable_files:
            file_path = os.path.join(cat_dir, filename)
            try:
                # 특징 추출 (색상 히스토그램)
                feature = classifier.extract_features(file_path)
                X.append(feature)
                y.append(category)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return X, y

def train_test_split(X, y, test_ratio=0.2):
    """
    데이터를 무작위로 섞어서 훈련용(Train)과 평가용(Test)으로 분할합니다.
    (scikit-learn의 train_test_split 대체제)
    """
    combined = list(zip(X, y))
    random.shuffle(combined)
    
    split_index = int(len(combined) * (1 - test_ratio))
    
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
    
    if not os.path.exists(dataset_dir):
        print("경고: dataset_augmented 폴더가 없습니다. 1_augment_data.py 를 먼저 실행해주세요!")
        return
        
    X, y = load_dataset(dataset_dir)
    print(f"\n[Done] 총 {len(X)}개의 데이터 특징이 성공적으로 로드되었습니다.")
    
    # 80%는 훈련용(비교 원본), 20%는 평가용(풀어야 할 문제)으로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    print(f"훈련 데이터: {len(X_train)}개, 평가(테스트) 데이터: {len(X_test)}개\n")
    
    # KNN 모델 초기화 (k=5: 가장 가까운 5개의 이미지를 보고 다수결 결정)
    model = SimpleKNNClassifier(k=5)
    
    # 모델에 훈련 데이터 적재
    model.fit(X_train, y_train)
    
    print("\n[Start] 모델 평가를 시작합니다...")
    # 테스트 데이터에 대해 예측 수행
    predictions = model.predict(X_test)
    
    # 성능 평가 분석
    correct_count = 0
    category_stats = {
        'game': {'total': 0, 'correct': 0},
        'person': {'total': 0, 'correct': 0},
        'finance': {'total': 0, 'correct': 0}
    }
    
    for true_label, pred_label in zip(y_test, predictions):
        category_stats[true_label]['total'] += 1
        if true_label == pred_label:
            correct_count += 1
            category_stats[true_label]['correct'] += 1
            
    # 결과 터미널 UI 출력 (이 부분이 PPT 작성 시 좋은 자료가 됨)
    print("="*50)
    print(" [Report] 스냅소트(SnapSort) KNN 성능 평가 결과보고서 ")
    print("="*50)
    
    accuracy = (correct_count / len(y_test)) * 100
    print(f"▶ 종합 평균 인식률(Accuracy): {accuracy:.2f}% ({correct_count}/{len(y_test)})\n")
    
    print("▶ 카테고리별 인식률:")
    for cat, stats in category_stats.items():
        if stats['total'] > 0:
            cat_acc = (stats['correct'] / stats['total']) * 100
            print(f"  - {cat.upper()}: {cat_acc:.2f}% ({stats['correct']}/{stats['total']})")
        else:
            print(f"  - {cat.upper()}: 평가 데이터 없음")
    print("="*50)
    print("평가가 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main()
