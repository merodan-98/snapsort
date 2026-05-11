import os
import shutil
from PIL import Image, ImageEnhance
import random

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_image(image_path, output_dir, file_prefix):
    # 원본 이미지 복사하고 모자란 데이터 뻥튀기(Augmentation)하는 함수
    img = Image.open(image_path)
    
    # RGBA 이미지 들어가면 에러날까봐 일단 RGB로 다 변환함
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 1. 일단 원본 저장
    img.save(os.path.join(output_dir, f"{file_prefix}_original.jpg"))
    
    # 2. 좌우 반전해서 한장 더 추가
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(os.path.join(output_dir, f"{file_prefix}_flipped.jpg"))
    
    # 3. 색상/대비 약간 바꿔서 또 한장 추가 (UI 스샷이라 회전보단 이게 나을듯)
    enhancer = ImageEnhance.Contrast(img)
    contrast_img = enhancer.enhance(random.uniform(0.7, 1.3))
    contrast_img.save(os.path.join(output_dir, f"{file_prefix}_contrast.jpg"))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'dataset')
    target_dir = os.path.join(base_dir, 'dataset_augmented')
    
    if not os.path.exists(source_dir):
        print(f"에러: {source_dir} 폴더가 없어요.")
        return

    categories = ['game', 'person', 'finance']
    
    # 폴더 만들기
    create_directory(target_dir)
    for category in categories:
        create_directory(os.path.join(target_dir, category))
        
    print("--- 데이터 증강(Augmentation) 시작 ---")
    
    for category in categories:
        category_path = os.path.join(source_dir, category)
        target_category_path = os.path.join(target_dir, category)
        
        if not os.path.isdir(category_path):
            print(f"{category} 폴더 없어서 패스함")
            continue
            
        files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        print(f"\n[{category}] 원본 갯수: {len(files)}개")
        
        for i, filename in enumerate(files):
            file_path = os.path.join(category_path, filename)
            file_prefix = f"img_{i:04d}"
            
            # 금융(finance) 쪽 스샷이 너무 적어서 이것만 3배로 늘림
            if category == 'finance':
                augment_image(file_path, target_category_path, file_prefix)
            else:
                # 나머지는 갯수 충분하니까 그냥 복사만 (이름만 깔끔하게 맞춤)
                img = Image.open(file_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(os.path.join(target_category_path, f"{file_prefix}_original.jpg"))
                
        result_files = os.listdir(target_category_path)
        print(f"[{category}] 끝! 증강 완료된 갯수: {len(result_files)}장")

if __name__ == '__main__':
    main()
