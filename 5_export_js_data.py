import os
import json
import importlib.util

# 내가 만든 knn_classifier 불러오기
spec = importlib.util.spec_from_file_location("knn", "2_knn_classifier.py")
knn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn)
SimpleKNNClassifier = knn.SimpleKNNClassifier

def export_data_to_js(dataset_dir, output_file):
    print("[시스템] 프론트엔드용 정적 데이터 추출중 (이거 짱 편함)...")
    
    classifier = SimpleKNNClassifier()
    categories = ['game', 'person', 'finance']
    
    model_data = []
    
    for category in categories:
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir):
            continue
            
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        for filename in files:
            file_path = os.path.join(cat_dir, filename)
            try:
                # 파이썬 Numpy 배열을 통째로 자바스크립트 리스트로 바꿔서 꽂아넣음
                feature = classifier.extract_features(file_path).tolist()
                model_data.append({
                    "label": category,
                    "vector": feature
                })
            except Exception as e:
                print(f"파일 깨짐 ㅡㅡ 패스함: {filename}, 원인: {e}")
                
    # JS 파일로 걍 내보내버리기
    # 이러면 웹서버 없어도 그냥 index.html만 켜도 동작함 개꿀 ㅋㅋ
    js_content = f"const KNNDataset = {json.dumps(model_data)};\n"
    js_content += f"console.log('[완료] 프론트엔드에 총 ' + KNNDataset.length + '개 데이터 쏴줌');"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_content)
        
    print(f"다 뽑았당: 총 {len(model_data)}개 -> {output_file}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset_augmented')
    output_path = os.path.join(base_dir, 'export_web', 'model_data.js')
    
    export_data_to_js(dataset_dir, output_path)
