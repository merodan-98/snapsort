import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import importlib.util

# 1. 뼈대되는 내 모델 불러오기
spec = importlib.util.spec_from_file_location("knn", "2_knn_classifier.py")
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)
SimpleKNNClassifier = knn.SimpleKNNClassifier

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
CATEGORIES = ('game', 'person', 'finance')
CATEGORY_LABELS = {'game': '게임', 'person': '인물', 'finance': '금융'}


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def collect_image_paths(source_paths):
    """폴더·파일 경로 목록에서 이미지 파일 경로만 모음 (하위 폴더 포함)."""
    images = []
    seen = set()
    for path in source_paths:
        path = os.path.normpath(path)
        if os.path.isfile(path) and is_image_file(path):
            if path not in seen:
                seen.add(path)
                images.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    if is_image_file(filename):
                        full = os.path.normpath(os.path.join(root, filename))
                        if full not in seen:
                            seen.add(full)
                            images.append(full)
    return sorted(images)


def unique_dest_path(dest_dir, filename):
    """같은 이름이 있으면 photo_1.jpg 식으로 번호 붙임."""
    base, ext = os.path.splitext(filename)
    dest = os.path.join(dest_dir, filename)
    counter = 1
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{base}_{counter}{ext}")
        counter += 1
    return dest


def organize_images(model, image_paths, output_dir, mode='copy'):
    """
    이미지마다 KNN 분류 후 output_dir/{game|person|finance}/ 로 복사 또는 이동.
    반환: (성공 건수, 카테고리별 개수 dict, 실패 목록)
    """
    counts = {cat: 0 for cat in CATEGORIES}
    failed = []
    success = 0

    for cat in CATEGORIES:
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    for file_path in image_paths:
        try:
            feature = model.extract_features(file_path)
            label = model.predict([feature])[0]
            if label not in CATEGORIES:
                failed.append((file_path, f"알 수 없는 라벨: {label}"))
                continue

            dest_dir = os.path.join(output_dir, label)
            dest_path = unique_dest_path(dest_dir, os.path.basename(file_path))

            if mode == 'move':
                shutil.move(file_path, dest_path)
            else:
                shutil.copy2(file_path, dest_path)

            counts[label] += 1
            success += 1
        except Exception as e:
            failed.append((file_path, str(e)))

    return success, counts, failed


def get_evaluation_stats(dataset_dir):
    # 띄우기 전에 전체 데이터 모의평가 돌려서 정확도 뽑아오는 함수
    try:
        from train_loader import load_full_dataset
        import random
        X, y = load_full_dataset(dataset_dir)
        if not X:
            return None, 0, 0

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


class BatchOrganizeDialog(tk.Toplevel):
    """폴더·다중 파일을 선택해 카테고리별 하위 폴더로 정리하는 창."""

    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model
        self.source_paths = []
        self.output_dir = tk.StringVar()
        self.mode = tk.StringVar(value='copy')

        self.title("SnapSort - 갤러리 일괄 정리")
        self.geometry("520x420")
        self.configure(bg="#f0f0f0")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        tk.Label(
            self, text="갤러리 일괄 정리",
            font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333"
        ).pack(pady=(16, 4))

        tk.Label(
            self,
            text="사진을 game / person / finance 폴더로 자동 분류합니다.",
            font=("Helvetica", 10), bg="#f0f0f0", fg="#555"
        ).pack(pady=(0, 12))

        src_frame = tk.LabelFrame(self, text="1. 정리할 사진", bg="#f0f0f0", padx=12, pady=8)
        src_frame.pack(fill='x', padx=20, pady=4)

        btn_row = tk.Frame(src_frame, bg="#f0f0f0")
        btn_row.pack(fill='x')
        tk.Button(
            btn_row, text="폴더 선택", command=self.pick_folder,
            font=("Helvetica", 10), padx=8, pady=4
        ).pack(side='left', padx=(0, 8))
        tk.Button(
            btn_row, text="여러 파일 선택", command=self.pick_files,
            font=("Helvetica", 10), padx=8, pady=4
        ).pack(side='left')

        self.source_label = tk.Label(
            src_frame, text="선택된 항목 없음", font=("Helvetica", 9),
            bg="#f0f0f0", fg="#666", wraplength=440, justify='left'
        )
        self.source_label.pack(anchor='w', pady=(8, 0))

        out_frame = tk.LabelFrame(self, text="2. 정리 결과 저장 위치", bg="#f0f0f0", padx=12, pady=8)
        out_frame.pack(fill='x', padx=20, pady=8)

        out_row = tk.Frame(out_frame, bg="#f0f0f0")
        out_row.pack(fill='x')
        tk.Entry(out_row, textvariable=self.output_dir, width=42, font=("Helvetica", 9)).pack(
            side='left', fill='x', expand=True, padx=(0, 8)
        )
        tk.Button(
            out_row, text="찾아보기", command=self.pick_output,
            font=("Helvetica", 10), padx=8, pady=4
        ).pack(side='left')

        mode_frame = tk.LabelFrame(self, text="3. 처리 방식", bg="#f0f0f0", padx=12, pady=8)
        mode_frame.pack(fill='x', padx=20, pady=4)

        tk.Radiobutton(
            mode_frame, text="복사 (원본 유지)", variable=self.mode, value='copy',
            bg="#f0f0f0", font=("Helvetica", 10)
        ).pack(anchor='w')
        tk.Radiobutton(
            mode_frame, text="이동 (원본에서 제거)", variable=self.mode, value='move',
            bg="#f0f0f0", font=("Helvetica", 10)
        ).pack(anchor='w')

        self.progress_label = tk.Label(
            self, text="", font=("Helvetica", 9), bg="#f0f0f0", fg="#1565c0"
        )
        self.progress_label.pack(pady=(8, 0))

        self.run_btn = tk.Button(
            self, text="정리 시작", command=self.run_organize,
            font=("Helvetica", 12, "bold"), bg="#5E6AD2", fg="white",
            padx=24, pady=10
        )
        self.run_btn.pack(pady=16)

    def pick_folder(self):
        path = filedialog.askdirectory(title="정리할 사진이 있는 폴더")
        if path:
            self.source_paths = [path]
            self._update_source_label()

    def pick_files(self):
        paths = filedialog.askopenfilenames(
            title="정리할 이미지 선택",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")]
        )
        if paths:
            self.source_paths = list(paths)
            self._update_source_label()

    def pick_output(self):
        path = filedialog.askdirectory(title="정리된 사진을 저장할 폴더")
        if path:
            self.output_dir.set(path)

    def _update_source_label(self):
        if len(self.source_paths) == 1 and os.path.isdir(self.source_paths[0]):
            text = f"폴더: {self.source_paths[0]}"
        else:
            text = f"파일 {len(self.source_paths)}개 선택됨"
        images = collect_image_paths(self.source_paths)
        text += f"\n→ 이미지 {len(images)}장 발견"
        self.source_label.config(text=text)

    def run_organize(self):
        if not self.source_paths:
            messagebox.showwarning("안내", "정리할 폴더 또는 파일을 먼저 선택하세요.")
            return

        out = self.output_dir.get().strip()
        if not out:
            messagebox.showwarning("안내", "결과를 저장할 폴더를 선택하세요.")
            return

        image_paths = collect_image_paths(self.source_paths)
        if not image_paths:
            messagebox.showwarning("안내", "선택한 위치에서 이미지를 찾지 못했습니다.")
            return

        mode = self.mode.get()
        if mode == 'move':
            if not messagebox.askyesno(
                "이동 확인",
                f"이미지 {len(image_paths)}장을 원본 위치에서 옮깁니다.\n계속할까요?"
            ):
                return

        self.run_btn.config(state='disabled')
        self.progress_label.config(text="분류 및 폴더 정리 중... 잠시만 기다려 주세요.")
        self.update()

        try:
            success, counts, failed = organize_images(
                self.model, image_paths, out, mode=mode
            )
        except Exception as e:
            messagebox.showerror("에러", f"일괄 정리 중 오류: {e}")
            self.run_btn.config(state='normal')
            self.progress_label.config(text="")
            return

        self.run_btn.config(state='normal')
        self.progress_label.config(text=f"완료: {success}장 처리됨")

        lines = [
            f"총 {success}장을 아래 폴더에 정리했습니다.",
            f"저장 위치: {out}",
            "",
            "카테고리별:"
        ]
        for cat in CATEGORIES:
            n = counts[cat]
            if n:
                lines.append(f"  · {CATEGORY_LABELS[cat]} ({cat}): {n}장")

        if failed:
            lines.append(f"\n실패: {len(failed)}장")
            for path, err in failed[:5]:
                lines.append(f"  - {os.path.basename(path)}: {err}")
            if len(failed) > 5:
                lines.append(f"  ... 외 {len(failed) - 5}건")

        messagebox.showinfo("일괄 정리 완료", "\n".join(lines))

        if messagebox.askyesno("폴더 열기", "정리된 폴더를 탐색기에서 열까요?"):
            os.startfile(out)


class SnapSortGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SnapSort - 스마트폰 갤러리 정리기")
        self.root.geometry("600x760")
        self.root.configure(bg="#f0f0f0")

        # 화면 꾸미기
        title = tk.Label(
            root, text="SnapSort Image Classifier",
            font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333"
        )
        title.pack(pady=(16, 4))

        subtitle = tk.Label(
            root, text="AI 기반 갤러리 자동 분류 · 정리",
            font=("Helvetica", 11), bg="#f0f0f0", fg="#666"
        )
        subtitle.pack(pady=(0, 8))

        self.stats_label = tk.Label(
            root, text="모델 로딩중 (좀만 기다리셈)...",
            font=("Helvetica", 11), bg="#f0f0f0", fg="blue"
        )
        self.stats_label.pack(pady=5)

        btn_frame = tk.Frame(root, bg="#f0f0f0")
        btn_frame.pack(pady=12)

        self.upload_btn = tk.Button(
            btn_frame, text="이미지 1장 분류", command=self.upload_image,
            font=("Helvetica", 12), bg="#4caf50", fg="white", padx=16, pady=10
        )
        self.upload_btn.pack(side='left', padx=6)

        self.batch_btn = tk.Button(
            btn_frame, text="갤러리 일괄 정리", command=self.open_batch_dialog,
            font=("Helvetica", 12), bg="#5E6AD2", fg="white", padx=16, pady=10
        )
        self.batch_btn.pack(side='left', padx=6)

        # 여기다 이미지 띄울거임
        self.image_label = tk.Label(root, bg="#ddd", width=60, height=20)
        self.image_label.pack(pady=10)

        # 결과 나오는 텍스트
        self.result_label = tk.Label(
            root, text="한 장 분류 또는 일괄 정리를 선택하세요.",
            font=("Helvetica", 16, "bold"), bg="#f0f0f0"
        )
        self.result_label.pack(pady=12)

        hint = tk.Label(
            root,
            text="일괄 정리: 폴더/여러 파일 → game · person · finance 하위 폴더로 복사 또는 이동",
            font=("Helvetica", 9), bg="#f0f0f0", fg="#888", wraplength=520
        )
        hint.pack(pady=(0, 12))

        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, 'dataset_augmented')

        self.model, self.accuracy, self.total_data = get_evaluation_stats(dataset_dir)

        if self.model:
            self.stats_label.config(
                text=f"✅ 총 데이터: {self.total_data}장 | 평균 정확도: {self.accuracy:.2f}%"
            )
        else:
            self.stats_label.config(
                text="⚠️ 모델 구동 실패함.. 데이터셋 폴더 있나 확인 ㄱㄱ", fg="red"
            )
            self.batch_btn.config(state='disabled')

    def open_batch_dialog(self):
        if not self.model:
            messagebox.showerror("에러", "모델 아직 로딩 안됐음")
            return
        BatchOrganizeDialog(self.root, self.model)

    def upload_image(self):
        if not self.model:
            messagebox.showerror("에러", "모델 아직 로딩 안됐음")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")]
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, width=400, height=400)
            self.image_label.image = photo

            feature = self.model.extract_features(file_path)
            predictions = self.model.predict([feature])
            result = predictions[0]

            colors = {"game": "#F43F5E", "finance": "#10B981", "person": "#F59E0B"}
            color = colors.get(result, "#333")
            label_ko = CATEGORY_LABELS.get(result, result)
            self.result_label.config(
                text=f"분류 결과: [ {label_ko} / {result.upper()} ]", fg=color
            )

        except Exception as e:
            messagebox.showerror("에러", f"이미지 처리하다 터짐: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = SnapSortGUI(root)
    root.mainloop()
