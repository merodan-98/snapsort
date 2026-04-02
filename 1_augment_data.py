import os
import shutil
from PIL import Image, ImageEnhance
import random

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_image(image_path, output_dir, file_prefix):
    """
    원본 이미지를 복사하고, 증강(Augmentation)된 이미지를 추가로 생성합니다.
    """
    img = Image.open(image_path)
    # RGBA 등의 이미지일 수 있으니 RGB로 안전하게 변환
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 1. 원본 저장
    img.save(os.path.join(output_dir, f"{file_prefix}_original.jpg"))
    
    # 2. 좌우 반전
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(os.path.join(output_dir, f"{file_prefix}_flipped.jpg"))
    
    # 3. 색상/대비 약간 강화 (UI/테스트 화면이므로 회전보다는 대비/밝기를 조절해 텍스트 다양성을 줌)
    enhancer = ImageEnhance.Contrast(img)
    contrast_img = enhancer.enhance(random.uniform(0.7, 1.3))
    contrast_img.save(os.path.join(output_dir, f"{file_prefix}_contrast.jpg"))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'dataset')
    target_dir = os.path.join(base_dir, 'dataset_augmented')
    
    if not os.path.exists(source_dir):
        print(f"오류: {source_dir} 폴더가 존재하지 않습니다.")
        return

    categories = ['game', 'person', 'finance']
    
    # 출력된 폴더 구조 생성
    create_directory(target_dir)
    for category in categories:
        create_directory(os.path.join(target_dir, category))
        
    print("--- 데이터 증강 작업을 시작합니다 ---")
    
    for category in categories:
        category_path = os.path.join(source_dir, category)
        target_category_path = os.path.join(target_dir, category)
        
        if not os.path.isdir(category_path):
            print(f"경고: {category} 폴더가 없어 넘어갑니다.")
            continue
            
        files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        print(f"\n[{category}] 원본 데이터 수: {len(files)}장")
        
        for i, filename in enumerate(files):
            file_path = os.path.join(category_path, filename)
            file_prefix = f"img_{i:04d}"
            
            # finance일 경우에만 증강(Augmentation) 수행
            if category == 'finance':
                # 1장 -> 3장 (원본 1, 반전 1, 대비조절 1)
                augment_image(file_path, target_category_path, file_prefix)
            else:
                # 나머지는 단순 복사 (이름 통일)
                img = Image.open(file_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(os.path.join(target_category_path, f"{file_prefix}_original.jpg"))
                
        # 증강 완료 후 파악
        result_files = os.listdir(target_category_path)
        print(f"[{category}] 완료! 증강 후 데이터 수: {len(result_files)}장")

if __name__ == '__main__':
    main()
