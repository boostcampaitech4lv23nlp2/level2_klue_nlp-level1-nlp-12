
![header](https://capsule-render.vercel.app/api?type=transparent&animation=scaleIn&section=header&text=NLP%20ET&fontSize=70&desc=✨NLP_Team_12✨%20&descAlignY=80)

# 1. 프로젝트 개요   

<aside>
💡 Competitions : [NLP] 문장 내 개체간 관계 추출   
문장의 단어(Entitiy)에 대한 속성관 관계를 예측하는 인공지능 만들기

</aside>

### TimeLine
![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/696155c5-3b40-4039-92d7-d17771eed133/Untitled.png)

### 협업 방식
> **Notion**
> 
- Team Notion에 실험 결과 기록
- Kanban Board로 담당자 및 진행 상황 공유   
![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aee0d5f8-4b31-42a3-abc8-210e5367913f/Untitled.png)   

> **Git**
> 
- Git Commit Message Convention
    - feat : 새로운 기능 추가
    - fix : 버그 수정
    - docs : 문서 수정
    - style : 코드 포맷팅, 세미콜론 누락, 코드 변경이 없는 경우
    - refactor : 코드 리펙토링
    - test : 테스트 코드, 리펙토링 테스트 코드 추가
    - chore : 빌드 업무 수정, 패키지 매니저 수정
    - <참고> 🔗 [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.2/#specification)
- Git flow
    - master : 제품으로 출시될 수 있는 브랜치
    - develop : 다음 출시 버전을 개발하는 브랜치
    - feature : 기능을 개발하는 브랜치
- Pre-commit
    - CI/CD - black, isort, autoflake → flake8
- Git hub action
    - Pre-commit : flake8
    - Commit Convention
   
# 2. 프로젝트 팀 구성 및 역할   
🔬**EDA** : 용찬

> Exploratory Data Analysis, Reference searching
>

🗂️ **Data** : 건우, 단익

> Data Experiment, searching the pre-trained models
>

🧬 **MODEL** : 재덕, 석희

> to reconstruct the baseline, searching the pre-trained models
>

# 3. 프로젝트 수행 절차 및 방법
## 1) EDA

### a. 문제 정의

- Entity의 위치가 embedding size인 512를 넘어가는 데이터 확인
    - Embedding size를 넘어가는 데이터 drop
- 한자가 포함되어 있는 데이터 확인
    - hanja 라이브러리를 통한 한자-한국어 변환
- Baseline preprocessing 함수 오류 확인
- Entity type별 편향 확인¹⁾
    - Entity type별 class restriction 실험

### b. 데이터 시각화

- Sentence 문장 길이 시각화
    - Train, test data 분포가 동일한 것 확인
- Label 분포 시각화
- Source column 시각화
    - Wikipedia, wikitree, policy_briefing 분포 확인
    - 문어체 데이터로 pre-training한 모델에 집중

### c. An Improved Baseline for Sentence-level Relation Extraction²⁾

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/23747f42-8c04-4c5b-b833-0798a327b0a5/Untitled.png)

위 다섯 가지 실험 모두 실행한 결과 성능 개선을 이루어내지는 못했습니다.

## 2) Model

### a. Pytorch Lightning refactoring

- Pytorch Lightning 이식 및 실험 지원

### b. Training Config

- Optimizer
    - AdamW
    - weight decay : Overfitting을 방지하기 위해 추가
    - LR Scheduler
        - constant_warmup : 정해진 step까지 LR이 선형적으로 증가하며 이후 고정된 값으로 학습
        - cosine_warmup : warm up 과정 이후 cosine 함수를 통해 LR scaling 수행
    - LR finder : Pytorch Lightining의 lr_finder 기능을 통해 초기 LR 설정³⁾

![Learning rate finder](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/53364874-b9c6-453e-9282-de81eb61c0c1/Untitled.png)

Learning rate finder

- Loss Function
    - CrossEntropy : (baseline loss) input logits과 target 사이 cross entropy loss 계산
    - Focal Loss : class imbalance dataset에서 class 별 가중치를 loss에 반영하기 위해 사용⁴⁾
    - Label Smoothing Loss : hard target을 soft target으로 바꾸어 모델의 over confidence문제를 개선하기 위해 사용
    - F1 Loss : classification loss 계산하기 위해 사용
- seed_everything & deterministic을 사용하여 재현 보장
- Mixed precision : GPU resource를 효율적으로 사용하며 연산속도 증가
- Stratified KFold : 불균형한 데이터셋이기에 모델 성능 일반화를 위해 사용
- Batch Size Finder : largest batch size를 찾는 기능

![cosine warmup](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4729728f-303c-44d4-ab52-52a22719eee2/Untitled.png)

cosine warmup

- 실험 결과, LR Scheduler 중 **cosine warmup** 을 사용했을 때 성능향상을 보였습니다. 이는 schduler가 초기에 워밍업 기간을 가짐으로써, 모델이 early over-fitting되는 것을 방지할 수 있었습니다.

### c. 논문 구현

- R-Roberta : R-bert 매커니즘을 Roberta에 적용⁵⁾
- CoRE: Counterfactual Analysis based Relation Extraction⁶⁾

## 3) ****Data Experiments****

### a. Data Augmentation

- **Data pre-processing :** EDA를 통해 한자, 영문자 등의 문자 비율이 많은 것을 확인했고, 한자를 (한자)라는 형태로 전처리하여 데이터 성능을 비교하였습니다. 한자 전처리한 모델(roberta-base_hanja_16_1e-05) : test_auprc 0.465 향상되었으나 test_accuracy,test_f1는 원본 데이터 모델이 더 높은 성능을 갖고 있어 위 실험은 제외했습니다.
- **Back Translation :** 한-영-한 역번역을 시도하려고 하였으나, EDA를 통해 대부분의 영문자를 포함한 단어는 고유명사로 확인되어 한-일-한 역번역을 진행했습니다. 역번역 데이터 20,011 rows를 추가하였지만, 원본 데이터의 성능이 더 높게 나와 위 실험을 제외했습니다.
- **EDA(easy data augmentation)**를 시도하려고 하였으나, 이전 기수에서 실패한 실험으로 확인되어 제외했습니다. 특히, SR(동의어 교체)를 시도하였으나 교체할 entity 단어는 지명, 이름, 고유명사 단어가 많아서 유의어 교체할 데이터가 상대적으로 적은 편으로 확인되어 제외했습니다.
- **Generation Model** : 생성 모델(koGPT3)을 사용하여, sub-obj entity를 활용한 새로운 문장을 구현하는 하는 테스트를 진행했습니다. 하지만 두 entity관계를 제대로 나타내는 문장을 제대로 구현하지 못해 위 실험은 제외했습니다.
- **Masked Language Model** : 문장에 sub,obj entity 부분을 [MASK]로 처리하고 bert 모델로 [MASK]의 새로운 단어를 찾는 실험을 진행했습니다. 기존 entity와 다른 단어를 생성하게 하여 새로운 entity를 추가하였습니다. 증강 데이터 58,446 rows를 추가한 실험 결과, 원본 데이터보다 향상된 성능을 확인하지 못하여 위 실험은 제외되었습니다.

### b. 논문 구현

- **Unipelt, Lora** : finetuning 시 전체 parmeter 를 학습하는 것이 아닌 추가적으로 학습가능한 파라미터를 모델에 추가하여 학습의 효율성을 향상시키는 실험을 진행하고자 했습니다. 하지만, 모델 수정에 어려움이 있어 구현을 완료하지 못했습니다.

## 4) Optimization

### a. ELECTRA, RoBERTa, R-RoBERTa, BigBird 등 다양한 PLM에 대한 최적화 수행

- **batch size**:16
**loss**: CrossEntropy loss, Label smoothing
**learning rate scheduler**: cosine warmup scheduler, constant warmup scheduler
**Initial learning rate**: LR finder 기반 모델별 최적 learning rate 사용 (ex. roberta-large: 2.12e-05)

## 5) Ensemble

### a. Soft Voting

- 모델별 micro-f1 score를 기준으로 가중 평균을 구하는 방식으로 구현했습니다.
- ELECTRA(3개), RoBERTa(4개), R-RoBERTa(2개)를 앙상블한 결과, micro-f1 74.1204, auprc 77.1611 성능이 가장 좋았습니다.

### b. Hard Voting

- 예측된 라벨값인 pred_label을 기준으로 최빈값을 도출하는 방식으로 구현했습니다.
- Soft Voting된 probs를 유지하고, 앙상블된 ELECTRA, RoBERTa, R-RoBERTa, BigBird를 앙상블한 결과, micro-f1 74.4339, auprc 77.1611로 성능이 가장 좋았습니다.

# 4. 프로젝트 수행 결과

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/05467360-1efe-4365-88af-3417a6a31906/Untitled.png)

- 최종 결과 14 위 (14/14), F1 기준 71.97 획득

# 5. 결론

### PL사용, 논문 참고 등을 통해 다방면으로 코드 개선을 수행했으며, 데이터 분석 및 실험, 앙상블 과정을 통해 최종 결과를 도출

- 데이터 분석을 통한 데이터 품질 개선(data cleaning, data augmentation)
- 데이터셋에 적한한 PLM 선정 및 최적화
- 베이스라인 코드 개선
    - PL 이식 및 다양한 기능 추가
- 논문 구현을 통한 추가적인 모델 개선을 수행
    - R-roberta, CoRE, Unipelt 등의 논문을 참고 및 구현을 위한 실험 수행
- 다양한 결과에 대한 앙상블(Hard Voting)을 수행

# 6. Future Study

### 추가적인 개선 방향

- 데이터 불균형에 따른 개선 방법 필요
    - focal loss 등을 사용하였으나, 개선되지 않음
    - under/over sampling에 대한 추가 실험 필요
- Prompt를 활용하여 데이터 input을 변경하는 실험 필요⁷⁾
    - Baseline: sub_entity [SEP] obj_entity [SEP] sentence
    Prompt: [CLS] sub_entity와 obj_entity의 관계는? [SEP] sentence

# 7. Appendix

- RoBERTa-Large Loss function 선택

| Model | Loss function | Learning Rate | Epoch | Batch Size | F1 score | AUPRC | inference / F1 | inference / AUPRC |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RoBERTa-Large | CrossEntropy | 2.1216154368926846e-05 | 2 | 16 | 84.009 | 80.142 | 69.66 | 74.39 |  |
|  | F1 loss | 2.1216154368926846e-05 |  | 16 | 60.035 | 26.119 |  |  |  |
|  | Label smoothing | 2.1216154368926846e-05 | 1 | 16 | 84.131 | 78.105 | 68.0612 | 70.4386 |  |
|  | Focal | 2.1216154368926846e-05 | 2 | 16 | 83.403 | 77.717 |  |  |  |
|  | CrossEntropy | 2.1216154368926846e-05 |  | 16 | 84.178 | 80.415 |  |  | scheduler |
|  | CrossEntropy | 2.1216154368926846e-05 | 각 2 epoch | 16 |  |  | 66.0331 | 69.9273 | k-fold(5) |
|  | CrossEntropy  | 2.1216154368926846e-05 |  | 16 | 84.783 | 80.62 |  |  | scheduler -cosine warmup |
|  | CrossEntropy | 2.1216154368926846e-05 | 2 | 16 | 90.85 | 91.093 | 61.2271 | 63.5467 | augmentation  entities |

| Model | Loss function | Learning Rate | Scheduler | Batch Size | Data | F1 score | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electra | CrossEntropy | 4.487453899331321e-05 |  | 16 | Original | 81.077 | 74.163 |
|  | Label smoothing | 4.487453899331321e-05 |  | 16 | Original | 80.552 | 71.444 |
|  | CrossEntropy | 4.487453899331321e-05 |  | 16 | Multi label Augmentation | 80.248 | 65.376 |
|  | Label smoothing | 4.487453899331321e-05 |  | 16 | Multi label Augmentation | 80.003 | 67.314 |
|  | CrossEntropy | 4.487453899331321e-05 | cosine_warmup | 16 | Original | 81.213 | 72.594 |
|  | Label smoothing | 4.487453899331321e-05 | cosine_warmup | 16 | Original | 80.418 | 70.923 |
|  | CrossEntropy | 4.487453899331321e-05 | cosine_warmup | 16 | Multi label Augmentation | 80.003 | 67.314 |
|  | Label smoothing | 4.487453899331321e-05 | cosine_warmup | 16 | Multi label Augmentation | 79.838 | 66.963 |
|  | Label smoothing |  | cosine_warmup |  | Original | 84.735 | 81.839 |
|  | CrossEntropy |  | cosine_warmup |  | Multi label Augmentation | 85.2134 | 80.60365 |
| RoBERTa-Large | Label smoothing |  | constant_warmup |  | Multi label Augmentation | 83.61 | 80.239 |
|  | Label smoothing |  | cosine_warmup |  | Multi label Augmentation | 83.61 | 80.239 |
| koBigbird | CrossEntropy |  | cosine_warmup |  | Original | 83.456 | 77.203 |
|  | Label smoothing |  | cosine_warmup |  | Original | 83.144 | 75.198 |
| RoBERTa-Large | CrossEntropy |  | cosine_warmup |  | Multi label Augmentation | 83.913 | 80.28 |
| RoBERTa-Large
(1 epoch) | CrossEntropy |  | cosine_warmup |  | Multi label Augmentation | 82.828 | 75.403 |
| koBigbird | CrossEntropy |  |  |  | Multi label Augmentation |  |  |
|  | Label smoothing |  |  |  | Multi label Augmentation |  |  |
|  | Focal |  |  |  | Multi label Augmentation |  |  |
| R-ROBERTa-Large | CrossEntropy | 1.3182567385564076e-05 | cosine_warmup | 16 | Multi label Augmentation | 83.913 | 80.28 |
| R-ROBERTa-Large | CrossEntropy | 1.3182567385564076e-05 | cosine_warmup | 16 | Original | 84.491 | 80.874 |

---

1. [Relation Classification with Entity Type Restriction](https://arxiv.org/pdf/2105.08393.pdf) 
2. [An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/pdf/2102.01373.pdf)
3. [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)
4. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002v2) - “Focal Loss **focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training.”
5. [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284) - R-RoBERTa 구현 참고
6. [Should We Rely on Entity Mentions for Relation Extraction? Debiasing Relation Extraction with Counterfactual Analysis](https://arxiv.org/pdf/2205.03784.pdf) - “CoRE (Counterfactual Analysis based Relation Extraction) debiasing method that guides the RE models to focus on the main effects of textual context without losing the entity information”
7. [PTR: Prompt Tuning with Rules for Text Classification](https://arxiv.org/pdf/2105.11259.pdf)
