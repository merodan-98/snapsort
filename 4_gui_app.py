import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import importlib.util

# 1. 모델과 핵심 헬퍼 로드
spec = importlib.util.spec_from_file_location("knn", "2_knn_classifier.py")
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)
SimpleKNNClassifier = knn.SimpleKNNClassifier

def get_evaluation_stats(dataset_dir):
    """서버 구동 시와 마찬가지로 모의 평가를 통해 성능(Accuracy)을 알아옵니다."""
    try:
        from train_loader import load_full_dataset
        import random
        X, y = load_full_dataset(dataset_dir)
        if not X: return None, 0, 0
        
        combined = list(zip(X, y))
        random.seed(42)
        random.shuffle(combined)
        split_index = int(len(combined) * 0.8)
        
        train_data, test_data = combined[:split_index], combined[split_index:]
        X_train, y_train = [i[0] for i in train_data], [i[1] for i in train_data]
        X_test, y_test = [i[0] for i in test_data], [i[1] for i in test_data]
        
        model = SimpleKNNClassifier(k=5)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        correct = sum(1 for t, p in zip(y_test, predictions) if t == p)
        accuracy = (correct / len(y_test)) * 100
        return model, accuracy, len(X)
    except Exception as e:
        print(e)
        return None, 0, 0

class SnapSortGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SnapSort - 순수 파이썬 데스크톱 시연기")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f0f0")
        
        # UI 레이아웃 설정
        title = tk.Label(root, text="SnapSort Image Classifier", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=20)
        
        self.stats_label = tk.Label(root, text="모델 로딩 중...", font=("Helvetica", 11), bg="#f0f0f0", fg="blue")
        self.stats_label.pack(pady=5)
        
        # 이미지 업로드 버튼
        self.upload_btn = tk.Button(root, text="이미지 파일 열기", command=self.upload_image, font=("Helvetica", 12), bg="#4caf50", fg="white", padx=20, pady=10)
        self.upload_btn.pack(pady=20)
        
        # 이미지 표시 영역
        self.image_label = tk.Label(root, bg="#ddd", width=60, height=20)
        self.image_label.pack(pady=10)
        
        # 결과 표시 라벨
        self.result_label = tk.Label(root, text="이미지를 업로드하면 결과가 나옵니다.", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
        self.result_label.pack(pady=20)
        
        # 실제 모델 데이터 불러오기 (비동기 대신 직접 로드)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, 'dataset_augmented')
        
        self.model, self.accuracy, self.total_data = get_evaluation_stats(dataset_dir)
        
        if self.model:
            self.stats_label.config(text=f"✅ 총 데이터: {self.total_data}장 | 개발 프로그램 성능(평균 인식률): {self.accuracy:.2f}%")
        else:
            self.stats_label.config(text="⚠️ 모델 구동 실패. dataset_augmented 폴더를 확인하세요.", fg="red")

    def upload_image(self):
        if not self.model:
            messagebox.showerror("오류", "모델이 초기화되지 않았습니다.")
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if not file_path:
            return
            
        try:
            # 1. 이미지 화면에 띄우기
            img = Image.open(file_path)
            img.thumbnail((400, 400)) # 화면 비율에 맞게 리사이징
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, width=400, height=400)
            self.image_label.image = photo # 가비지 컬렉터 방지용 참조 저장
            
            # 2. 특징 추출 및 예측
            feature = self.model.extract_features(file_path)
            predictions = self.model.predict([feature])
            result = predictions[0]
            
            # 3. 색상 및 텍스트 표시
            colors = {"game": "#F43F5E", "finance": "#10B981", "person": "#F59E0B"}
            color = colors.get(result, "#333")
            self.result_label.config(text=f"분류 결과: [ {result.upper()} ]", fg=color)
            
        except Exception as e:
            messagebox.showerror("오류", f"이미지 처리 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SnapSortGUI(root)
    root.mainloop()
