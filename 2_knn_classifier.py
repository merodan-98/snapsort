import os
import math
import numpy as np
from PIL import Image

class SimpleKNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.training_data = []  # (feature_vector, label) 형태의 튜플 리스트
        self.labels = []

    def extract_features(self, image_path):
        """
        Numpy와 Pillow만을 사용하여 이미지에서 특징(Color Histogram)을 추출합니다.
        딥러닝 모델 대신 사용되는 핵심 알고리즘 파트입니다.
        """
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 1. 속도를 위해 이미지 해상도를 32x32로 대폭 축소
        # 형태나 질감보다는 '어떤 색이 많이 쓰였는가'를 중점적으로 봅니다.
        img = img.resize((32, 32))
        
        # 2. RGB 픽셀 데이터를 Numpy 배열로 변환
        pixel_data = np.array(img)
        
        # 3. 색상 히스토그램 추출 (R, G, B 각각 8개의 구간으로 나누어 빈도 계산)
        # 8 x 8 x 8 = 512차원의 특징 벡터 생성
        hist, _ = np.histogramdd(
            pixel_data.reshape(-1, 3),
            bins=(8, 8, 8),
            range=((0, 256), (0, 256), (0, 256))
        )
        
        # 4. 정규화 (Normalization) - 이미지 크기가 달라도 일관된 값을 가지도록
        feature_vector = hist.flatten()
        feature_vector = feature_vector / np.sum(feature_vector)
        
        return feature_vector

    def fit(self, X_train, y_train):
        """
        훈련 데이터를 모델(메모리)에 저장합니다. (KNN은 별도의 학습 과정이 없고 데이터 저장 자체가 학습입니다)
        """
        self.training_data = list(zip(X_train, y_train))
        print(f"KNN 모델에 총 {len(self.training_data)}개의 학습 데이터가 적재되었습니다.")

    def _euclidean_distance(self, vec1, vec2):
        """
        [수학 수식 직접 구현 파트] 유클리디안 거리(Euclidean Distance)
        두 특징 벡터 간의 거리를 넘파이 배열 연산으로 구합니다.
        """
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def predict(self, X_test):
        """
        새로운 데이터가 들어왔을 때 가장 가까운 상위 K개를 찾아 다수결로 카테고리를 추론합니다.
        """
        predictions = []
        for test_vec in X_test:
            # 1. 테스트 이미지와 모든 학습 이미지 간의 거리를 계산
            distances = []
            for train_vec, label in self.training_data:
                dist = self._euclidean_distance(test_vec, train_vec)
                distances.append((dist, label))
            
            # 2. 거리가 짧은 순서대로 정렬 (가장 비슷한 이미지 찾기)
            distances.sort(key=lambda x: x[0])
            
            # 3. 상위 K개의 이웃 선택
            k_nearest_neighbors = distances[:self.k]
            
            # 4. 다수결 투표 알고리즘 (어느 클래스가 가장 많이 나왔나?)
            votes = {}
            for dist, label in k_nearest_neighbors:
                votes[label] = votes.get(label, 0) + 1
                
            # 가장 투표를 많이 받은 클래스 선택
            majority_class = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0][0]
            predictions.append(majority_class)
            
        return predictions

# 단독 실행 시 테스트용 더미 코드
if __name__ == '__main__':
    print("이 스크립트는 모듈입니다. 3_evaluate.py에서 불러와 사용합니다.")
