## 프로젝트명
이커머스 리뷰 NLP 분석

## 프로젝트 개요
리뷰 데이터를 사용하여 다국어 모델 LoRA 파인튜닝
AI 모델링을 통해 고객 만족도 정밀 분류

## 핵심 성과: 
Baseline 대비 Weighted F1 Score 약 3배 향상 (0.19 → 0.61)

## 주요 기능(특징)
* 다국어 감성 분석: 포르투갈어 기반 리뷰 처리를 위해 huggingface에서 다국어 모델(`tabularisai/multilingual-sentiment-analysis`)을 베이스로 활용
* 클래스 불균형 처리: compute_class_weight를 활용하여 각 클래스(별점 1~5)에 가중치를 부여, 데이터 불균형 문제(별점 5점 편중)를 완화하고 소수 클래스의 학습 효율을 높였습니다.
* 효율적 파인튜닝 (PEFT): LoRA(Rank=8, Alpha=16)를 적용하여 전체 파라미터를 학습하는 대신 적은 파라미터로 모델을 최적화했습니다.
* 심층 오류 분석 (Error Analysis):
    * Confidence 기반 분석: 예측 확률(Confidence)과 차이(Gap)를 계산하여 모델이 확신하지 못하는 '애매한 샘플'을 식별합니다.
    * 액션 태그 자동화: 오분류 유형에 따라 '재라벨링 후보', '전처리 개선', '규칙 보완 후보' 등의 태그를 자동으로 부여하여 후속 조치를 제안합니다.
* 키워드 분석: 부정 리뷰(Class 0, 1)에서 자주 등장하는 키워드(ex: 'produto', 'recebi', 'veio')를 추출하여 문제점을 파악합니다.

## Model Configuration
* Base Model: tabularisai/multilingual-sentiment-analysis(BERT based)
* Tokenizer: AutoTokenizer (max_length: 128)
* Adapter: LoRA (Target modules: q_lin, v_lin, Rank: 8, Dropout: 0.1)
* Loss Function: Weighted CrossEntropyLoss (클래스별 가중치 적용)
* Training Strategy: Custom Trainer(WeightedTrainer)를 사용하여 Class Weight가 적용된 Loss를 계산합니다.

## 개발내용
1. 환경 설정: 필수 라이브러리(bitsandbytes, peft 등) 설치 및 한글 폰트(NanumGothic) 설정.
2. 데이터 전처리: 리뷰 점수 매핑, Train/Val/Test 분할 (7:1.5:1.5), 토큰 길이 분석 및 max_length=128 설정.
   Train: 23,443건 (70%)
   Validation: 5,024건 (15%)
   Test: 5,024건 (15%)
4. Baseline 평가: 파인튜닝 전 사전 학습 모델의 성능 측정 (Weighted F1: 0.1891).
5. 모델 학습 (LoRA): Balanced 클래스 가중치를 계산하여 WeightedTrainer에 적용하고 LoRA 어댑터를 학습.
6. 성능 평가: Test 셋을 활용한 최종 성능 측정 및 Baseline과 비교, Confusion Matrix 시각화.
7. 인사이트 도출: 예측 실패 사례에 대한 Confidence 분석, 액션 태그 부여 및 부정 리뷰 키워드 추출.

## 결과 및 성과
* 성능 대폭 향상:
    * Weighted F1 Score: 0.1891 (Baseline) → 0.6149 (LoRA) 로 약 225% 향상되었습니다.
    * Class 4 (Very Positive) Recall: 8.9% → 84.18% 로 대폭 개선되어 실제 긍정 리뷰를 효과적으로 감지합니다.
* 인사이트:
    * 전체적으로 실사용 가능한 수준의 성능(Weighted F1 0.615)을 달성했으나, 소수 클래스(Neutral 등)에 대한 예측은 여전히 어려움(Macro F1 0.429)을 확인했습니다.
    * 주요 부정 키워드는 'produto'(제품), 'recebi'(받았다), 'veio'(왔다) 등으로 배송 및 제품 상태와 연관됨을 파악했습니다.

## 데이터
* 이름: Olist Brazilian E-Commerce Public Dataset
* 출처: Kaggle
* 행수: 33,491건
* 사용 컬럼:
 review_comment_message: 고객 리뷰 텍스트 (Input)
 review_score: 고객 만족도 점수 1~5점 (Target)
 product_category_name: 제품 카테고리 (분석용 보조 데이터)
* 타겟 레이블: review_score 를 04 감성 레이블로 매핑
    ◦ 1점 → 0 (Very Negative)
    ◦ 2점 → 1 (Negative)
    ◦ 3점 → 2 (Neutral)
    ◦ 4점 → 3 (Positive)
    ◦ 5점 → 4 (Very Positive)

## 트러블슈팅 
* 문제 1: 클래스 불균형으로 인한 편향**
    * **현상:** 데이터의 50% 이상이 5점인 탓에, 모델이 모든 데이터를 5점으로 예측하려는 경향 발생.
    * **해결:** `sklearn`을 이용해 클래스별 가중치를 계산하고, 이를 Loss Function에 반영하여 소수 클래스(부정 리뷰) 학습 비중을 높임.
* **문제 2: 애매한 리뷰 처리 (Ambiguity)**
    * **현상:** 긍정/부정이 모호한 리뷰에 대해 모델의 신뢰도가 낮음.
    * **해결:** 예측 확률 차이(Gap)가 0.1 미만인 샘플을 추출하는 로직을 구현. 이를 통해 '재라벨링'이 필요한 데이터를 선별하는 **Data Loop** 파이프라인 제안.
 
## 기술스택
| **Language** | Python 3.12+ |
| **Framework** | PyTorch, Hugging Face Transformers |
| **Fine-tuning** | PEFT (LoRA), bitsandbytes (Quantization), Accelerate |
| **Data Ops** | Pandas, NumPy, Datasets, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Infra** | Google Colab (GPU T4) |
