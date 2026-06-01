import os
import pickle
import datetime
import importlib.util

# 2_perceptron_classifier.py 임포트해옴
spec = importlib.util.spec_from_file_location("perceptron", "2_perceptron_classifier.py")
perceptron = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perceptron)
SimpleSLPClassifier = perceptron.SimpleSLPClassifier

def write_log(message, log_file):
    print(message)
    # 로그 파일에도 같이 쓰도록 함
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'perceptron_model.pkl')
    test_data_path = os.path.join(base_dir, 'test_dataset.pkl')
    log_file = os.path.join(base_dir, 'test_log.txt')
    
    # 이전 로그 남아있으면 지움
    if os.path.exists(log_file):
        os.remove(log_file)
        
    write_log(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 퍼셉트론 성능 평가 시작", log_file)
    
    # 모델 파일 없으면 에러내고 종료
    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        write_log("에러: perceptron_model.pkl 파일이 없음. train.py부터 먼저 돌려주세요.", log_file)
        return
        
    write_log("저장된 모델 데이터랑 테스트 데이터 불러오는 중...", log_file)
    
    # 학습용 특징 데이터와 라벨 불러오기
    with open(model_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
        
    with open(test_data_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
        
    write_log(f"로딩 끝: 학습용 {len(X_train)}개, 테스트용 {len(X_test)}개\n", log_file)
    
    # 단층 퍼셉트론 객체 만들고 학습 진행
    model = SimpleSLPClassifier()
    model.fit(X_train, y_train)
    
    write_log("\n테스트 셋으로 모델 정확도 평가 시작...", log_file)
    # 진짜 테스트 시작
    predictions = model.predict(X_test)
    
    # 결과 계산
    correct_count = 0
    category_stats = {
        'game': {'total': 0, 'correct': 0},
        'person': {'total': 0, 'correct': 0},
        'finance': {'total': 0, 'correct': 0}
    }
    
    # 내가 정답 맞췄는지 하나하나 비교
    for true_label, pred_label in zip(y_test, predictions):
        category_stats[true_label]['total'] += 1
        if true_label == pred_label:
            correct_count += 1
            category_stats[true_label]['correct'] += 1
            
    # 과제 제출용 결과 출력 (예쁘게 보이려고 선 넣음)
    write_log("="*50, log_file)
    write_log(" [결과 보고서] SnapSort 단층 퍼셉트론(SLP) 모델 성능 평가", log_file)
    write_log("="*50, log_file)
    
    accuracy = (correct_count / len(y_test)) * 100
    write_log(f"▶ 전체 평균 정확도(Accuracy): {accuracy:.2f}% 맞춤 ({correct_count}/{len(y_test)})\n", log_file)
    
    write_log("▶ 카테고리별로 보면:", log_file)
    for cat, stats in category_stats.items():
        if stats['total'] > 0:
            cat_acc = (stats['correct'] / stats['total']) * 100
            write_log(f"  - {cat.upper()}: {cat_acc:.2f}% ({stats['correct']}/{stats['total']})", log_file)
        else:
            write_log(f"  - {cat.upper()}: 테스트 데이터 부족함", log_file)
    write_log("="*50, log_file)
    write_log(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 성능 평가 끝!", log_file)

if __name__ == '__main__':
    main()
