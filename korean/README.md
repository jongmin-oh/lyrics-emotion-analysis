# 한국어 가사 감정 분류

## 가사 감정 분류 방법론 - 하이브리드 방법론

앞서서 설명했듯이 감정 분석(Sentiment Analysis)방법으로는 두 가지가 존재합니다.

1. Lexicon-based Approach - 사전 기반 접근 법
2. Machine Learning Approach - 기계 학습 기반 접근 법

하지만 두 가지 방법 모두 치명적인 단점을 가지고 있다.

사전 기반 방법은 단어만 보기 때문에 전체적인 문맥을 보지 못하는 단점이 있다.
 - 부정어를 처리하지 못하는 문제 "좋아하는 사람과 만날 수 없네요" -> 사랑으로 분류
 - 어떤 단어가 가장 중요한 단어인지 표현하지 못하는 문제 단어 중요도가 다 일정하다.

기계 학습 기반 방법은 분류해야하는 클래스가 많아 질 수록 정확도가 떨어지는 단점이 있다.
 - 특징을 추출해서 분류하는 것이기 때문에 각 클래서별 차별화된 특징이 있어야한다.
 - 감정에서 "행복"과 "사랑"에 대한 특징을 뚜렷하게 분류하기가 어렵다.
 - 학습데이터가 많아야 정확한 분류를 할 수 있다.

그래서 나는 한국어 가사 감정 분류 문제에서 두 가지 모델의 합친 하이브리드 모델을 구현했다.<br>
하이브리드 모델은 두 가지 모델의 단점을 극복하고 장점을 최대한 살린 모델이다.

## 하이브이드 방법론 : 머신러닝/딥러닝 기반 + 사전 기반

중간에 긍/부정 필터를 두어서 문장의 전체적인 문맥을 파악하고 <br>
필터를 거친 문장들은 각각의 단어들로 감정 정도를 판단하여 점수를 합산하는 모델 <br>

1) '사랑' 분류 사례
<img src = "https://user-images.githubusercontent.com/23625693/126859462-83373b8a-275c-445a-b623-25d28c979390.png" width="50%" height="50%">

2) '슬픔 분류 사례
<img src = "https://user-images.githubusercontent.com/23625693/126859477-7e793a5c-ff38-4470-bd50-0229de03c1af.png" width="50%" height="50%">


### 긍/부정 분류기 : BERT 이진 분류 모델

기존에 라벨링했던 데이터를 긍정적인 감정(1) 과 부정적인 감정(0) 으로 치환하여 학습

Transformer 패키지 
Keras 사용

위키피디아에서 사전 학습시킨 모델 다운로드
bert-base-multilingual-cased 사용 ( 104 languages , 12-layer , 768-hidden , 12-head, 110M parameters ) 

- Config
- Checkpoint
- Vocab.txt

Bert_tokenizer를
Vocab 기준으로 임베딩 : input token
두 개의 문장을 구분하는 : segment input
Self-Attention 의 입력 위치를 나타내는 : postion input
* 한국어의 경우에는 bert input을 제외하고 모두다 0을 사용함.

사전학습된 모델을 로드 BERT 모델 레이어 과정
1. 768차원으로 token , segment 임베딩
2. 12개의 셀프 어텐션 레이어
3. NSP , MLM
----------사전 학습 레이어----------------- 4. DropOut
5. Output : Dense( 1, sigmoid )
