import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import importlib.util

# 1. 뼈대되는 내 모델 불러오기
spec = importlib.util.spec_from_file_location("knn", "2_knn_classifier.py")
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)
SimpleKNNClassifier = knn.SimpleKNNClassifier

def get_evaluation_stats(dataset_dir):
    # 띄우기 전에 전체 데이터 모의평가 돌려서 정확도 뽑아오는 함수
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
        self.root.title("SnapSort - 순수 파이썬 데스크톱 시연기 (내가 만듦)")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f0f0")
        
        # 화면 꾸미기
        title = tk.Label(root, text="SnapSort Image Classifier", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=20)
        
        self.stats_label = tk.Label(root, text="모델 로딩중 (좀만 기다리셈)...", font=("Helvetica", 11), bg="#f0f0f0", fg="blue")
        self.stats_label.pack(pady=5)
        
        # 버튼 하나 만듦
        self.upload_btn = tk.Button(root, text="이미지 파일 열기", command=self.upload_image, font=("Helvetica", 12), bg="#4caf50", fg="white", padx=20, pady=10)
        self.upload_btn.pack(pady=20)
        
        # 여기다 이미지 띄울거임
        self.image_label = tk.Label(root, bg="#ddd", width=60, height=20)
        self.image_label.pack(pady=10)
        
        # 결과 나오는 텍스트
        self.result_label = tk.Label(root, text="이미지 올리면 결과 나옴.", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
        self.result_label.pack(pady=20)
        
        # 로딩 끝나면 정보 띄워줌
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, 'dataset_augmented')
        
        self.model, self.accuracy, self.total_data = get_evaluation_stats(dataset_dir)
        
        if self.model:
            self.stats_label.config(text=f"✅ 총 데이터: {self.total_data}장 | 내 프로그램 평균 정확도: {self.accuracy:.2f}%")
        else:
            self.stats_label.config(text="⚠️ 모델 구동 실패함.. 데이터셋 폴더 있나 확인 ㄱㄱ", fg="red")

    def upload_image(self):
        if not self.model:
            messagebox.showerror("에러", "모델 아직 로딩 안됐음")
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if not file_path:
            return
            
        try:
            # 1. 썸네일 만들어서 화면에 띄우기 (안그러면 원본 크기로 나와서 화면 뚫음)
            img = Image.open(file_path)
            img.thumbnail((400, 400)) 
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, width=400, height=400)
            self.image_label.image = photo # 가비지 컬렉터가 안날려먹게 꽉잡고있기
            
            # 2. 특징 뽑고 예측 돌리기
            feature = self.model.extract_features(file_path)
            predictions = self.model.predict([feature])
            result = predictions[0]
            
            # 3. 결과에 맞게 글자 색 바꿔서 이쁘게 보여줌
            colors = {"game": "#F43F5E", "finance": "#10B981", "person": "#F59E0B"}
            color = colors.get(result, "#333")
            self.result_label.config(text=f"분류 결과: [ {result.upper()} ]", fg=color)
            
        except Exception as e:
            messagebox.showerror("에러", f"이미지 처리하다 터짐: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SnapSortGUI(root)
    root.mainloop()
