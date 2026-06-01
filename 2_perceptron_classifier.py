import os
import numpy as np
from PIL import Image

class SimpleSLPClassifier:
    """
    수업 시간에 배운 단층 퍼셉트론(Single Layer Perceptron, SLP)과
    경사하강법(Gradient Descent)을 활용한 다중 클래스 이미지 분류기.
    (딥러닝 라이브러리 없이 순수 NumPy와 파이썬만으로 구현됨)
    """
    def __init__(self, learning_rate=0.05, epochs=150):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None  # 가중치 행렬 (Weights) - 입력 차원 x 출력 차원
        self.b = None  # 편향 벡터 (Biases) - 출력 차원
        self.classes = ['game', 'person', 'finance']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

    def extract_features(self, image_path):
        """
        Numpy와 Pillow만을 사용하여 이미지의 색상 히스토그램 특징 추출 (512차원).
        이 부분은 이미지의 시각적 형태보다는 색상 테마 분포를 파악하는 특징 추출기입니다.
        """
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 이미지 크기를 32x32로 줄여 처리 속도를 비약적으로 향상시킴
        img = img.resize((32, 32))
        pixel_data = np.array(img)
        
        # RGB 세 채널을 각각 8개 구간(bin)으로 쪼개서 총 8x8x8 = 512차원의 빈도(히스토그램) 생성
        hist, _ = np.histogramdd(
            pixel_data.reshape(-1, 3),
            bins=(8, 8, 8),
            range=((0, 256), (0, 256), (0, 256))
        )
        
        # 정규화 (전체 합이 1이 되도록 조정하여 이미지 해상도가 달라도 동일한 비율을 가지게 함)
        feature_vector = hist.flatten()
        sum_val = np.sum(feature_vector)
        if sum_val > 0:
            feature_vector = feature_vector / sum_val
        
        return feature_vector

    def fit(self, X_train, y_train):
        """
        [경사하강법 학습 핵심 로직]
        MNIST SLP 학습 이론과 100% 동일한 수식으로 가중치 W와 편향 b를 학습합니다.
        """
        X = np.array(X_train)  # 입력 데이터 행렬 (m, 512)
        y = np.array([self.class_to_idx[label] for label in y_train])  # 정답 인덱스 벡터 (m,)
        
        m, input_dim = X.shape
        num_classes = len(self.classes)
        
        # 1. 가중치(W)와 편향(b)의 초기화 (작은 난수로 시작)
        np.random.seed(42)  # 실행할 때마다 결과가 일관되게 고정되도록 시드 설정
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        
        # 2. 정답 라벨 원-핫 인코딩 (One-hot encoding)
        Y_oh = np.zeros((m, num_classes))
        Y_oh[np.arange(m), y] = 1.0
        
        print("\n" + "="*50)
        print(" [단층 퍼셉트론(SLP) 경사하강법 학습을 시작합니다] ")
        print("="*50)
        
        # 3. 에포크(Epoch)만큼 학습 반복 루프 돌리기
        for epoch in range(self.epochs):
            # (1) 순전파 (Forward Pass): Z = X * W + b
            Z = np.dot(X, self.W) + self.b
            
            # (2) 소프트맥스(Softmax) 활성화 함수 구현 (오버플로우 방지 처리 포함)
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
            
            # (3) 교차 엔트로피 손실 (Cross Entropy Loss) 계산
            loss = -np.mean(np.sum(Y_oh * np.log(probs + 1e-15), axis=1))
            
            # (4) 역전파 (Backpropagation) - 그레이디언트(기울기) 계산
            # dZ는 예측 확률분포와 실제 원핫 라벨의 오차 차이
            dZ = (probs - Y_oh) / m  # Shape: (m, num_classes)
            dW = np.dot(X.T, dZ)     # Shape: (input_dim, num_classes)
            db = np.sum(dZ, axis=0)  # Shape: (num_classes,)
            
            # (5) 경사하강법(Gradient Descent) 매개변수 업데이트
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
            # 현재 에포크 기준 학습 정확도(Accuracy) 계산
            preds_idx = np.argmax(probs, axis=1)
            train_acc = np.mean(preds_idx == y) * 100
            
            # 10 에포크마다 로그 출력
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d}/{self.epochs:03d} | 손실(Loss): {loss:.4f} | 학습 정확도: {train_acc:.2f}%")
        
        print("="*50)
        print("▶ 단층 퍼셉트론 모델 경사하강법 학습 완료!")
        print("="*50 + "\n")

    def predict(self, X_test):
        """
        학습 완료된 W와 b를 사용하여 새로운 이미지 특징에 대해 예측 결과를 반환합니다.
        """
        if self.W is None or self.b is None:
            raise ValueError("모델이 아직 학습되지 않았습니다. fit()을 먼저 호출해 주세요.")
            
        X = np.array(X_test)
        
        # 순전파 계산
        Z = np.dot(X, self.W) + self.b
        
        # 소프트맥스로 확률 예측
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        # 가장 높은 확률을 가진 뉴런의 인덱스 선택
        preds_idx = np.argmax(probs, axis=1)
        
        # 숫자를 한글/영어 카테고리 명으로 복원하여 반환
        return [self.idx_to_class[idx] for idx in preds_idx]

if __name__ == '__main__':
    print("이 스크립트는 모델 모듈입니다. train.py 나 GUI 앱 등에서 불러와 실행해야 합니다.")
