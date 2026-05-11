import os
import sys
import pickle

import importlib.util

# 내가 만든 knn_classifier 불러오기
spec = importlib.util.spec_from_file_location("knn", "2_knn_classifier.py")
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)
SimpleKNNClassifier = knn.SimpleKNNClassifier

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'knn_model.pkl')
    
    print("="*50)
    print(" [ SnapSort - 이미지 1장만 넣어서 예측해보기 ] ")
    print("="*50)
    
    # 학습된 모델 파일 없으면 튕겨냄
    if not os.path.exists(model_path):
        print("에러: knn_model.pkl 파일이 없음. 무조건 train.py부터 먼저 돌리세요!")
        return
        
    # pkl에서 모델 데이터 싹 다 가져오기
    with open(model_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
        
    # 모델 세팅
    model = SimpleKNNClassifier(k=5)
    model.fit(X_train, y_train)
    
    # 터미널에서 실행할 때 뒤에 경로 치면 그거 쓰고, 안치면 직접 치라고 띄워줌
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("테스트 해볼 이미지 경로 쳐주세요 (절대경로 추천):\n> ")
        
    # 윈도우에서 경로 복붙하면 따옴표 붙어서 에러나는거 방지용
    image_path = image_path.strip().strip('"').strip("'")
    
    if not os.path.exists(image_path):
        print(f"경로 잘못친듯요 파일이 없어요. 확인점: {image_path}")
        return
        
    print("\n분석 돌아가는 중...")
    try:
        # 하나만 뽑아서 예측 돌리기
        feature = model.extract_features(image_path)
        predictions = model.predict([feature])
        result = predictions[0]
        
        print("\n" + "="*50)
        print(f" ▶ 분류 결과: 이 이미지는 [ {result.upper()} ] 카테고리인거 같음!")
        print("="*50)
    except Exception as e:
        print(f"이미지 읽다가 터짐.. 에러내용: {e}")

if __name__ == '__main__':
    main()
