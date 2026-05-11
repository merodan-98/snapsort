import os
import math
import numpy as np
from PIL import Image

class SimpleKNNClassifier:
    def __init__(self, k=5):
        # 가까운 이웃 5개(k=5) 보고 결정함
        self.k = k
        self.training_data = []  # (특징 벡터, 라벨) 묶어서 저장할 리스트
        self.labels = []

    def extract_features(self, image_path):
        # Numpy랑 Pillow만 써서 색상 히스토그램 특징 뽑는 함수 (딥러닝 프레임워크 안씀!!)
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 1. 계산 빨리 하려고 이미지를 32x32로 팍 줄여버림
        # 모양보다는 색깔 분포가 중요해서 이렇게 해도 됨
        img = img.resize((32, 32))
        
        # 2. 픽셀 데이터를 Numpy 배열로 바꿈
        pixel_data = np.array(img)
        
        # 3. 색상 히스토그램 (RGB 각각 8구간으로 쪼개서 개수 셈 -> 8x8x8 = 512차원)
        hist, _ = np.histogramdd(
            pixel_data.reshape(-1, 3),
            bins=(8, 8, 8),
            range=((0, 256), (0, 256), (0, 256))
        )
        
        # 4. 정규화 (Normalization) -> 이미지 크기 달라도 비율로 맞춰줌
        feature_vector = hist.flatten()
        feature_vector = feature_vector / np.sum(feature_vector)
        
        return feature_vector

    def fit(self, X_train, y_train):
        # KNN은 학습 과정이 따로 없고 그냥 데이터 들고있는게 학습임
        self.training_data = list(zip(X_train, y_train))
        print(f"KNN 모델 학습 완료! (총 {len(self.training_data)}개 데이터 저장됨)")

    def _euclidean_distance(self, vec1, vec2):
        # [핵심 알고리즘] 유클리디안 거리 직접 수식으로 짠거
        # 넘파이로 두 벡터 사이 거리 계산함
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def predict(self, X_test):
        # 새 이미지 들어오면 저장해둔 데이터랑 거리 다 재보고 젤 가까운 애들 찾음
        predictions = []
        for test_vec in X_test:
            # 1. 모든 학습 이미지랑 거리 계산해서 리스트에 넣기
            distances = []
            for train_vec, label in self.training_data:
                dist = self._euclidean_distance(test_vec, train_vec)
                distances.append((dist, label))
            
            # 2. 거리 순으로 정렬 (짧은게 제일 비슷한거)
            distances.sort(key=lambda x: x[0])
            
            # 3. 위에서부터 K개만 짤라옴
            k_nearest_neighbors = distances[:self.k]
            
            # 4. 다수결 투표 (무슨 카테고리가 젤 많나?)
            votes = {}
            for dist, label in k_nearest_neighbors:
                votes[label] = votes.get(label, 0) + 1
                
            # 득표수 젤 많은 카테고리 고름
            majority_class = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0][0]
            predictions.append(majority_class)
            
        return predictions

if __name__ == '__main__':
    print("이건 모듈이라 그냥 실행하면 안되고 다른 스크립트에서 불러와서 써야됨!")
