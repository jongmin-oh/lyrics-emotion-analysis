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


## 긍/부정 분류기 : BERT 이진 분류 모델

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
3. NSP , MLM <br>
----------사전 학습 레이어----------------- 
4. DropOut
5. Output : Dense( 1, sigmoid )

<img src = "https://user-images.githubusercontent.com/23625693/126860642-b83afd87-3174-48bd-a101-66143eb8a2b0.png" width="30%" height="30%">

## 사전 기반 분류 모델 ( 긍/부정 필터 이후 분석 )

가사 도메인에 적합한 말뭉치 기반 감정 단어 사전 생성

8586개의 문장을 6개의 감정으로 빈도 수 별로 단어 사전 생성

### 단어 별 감정 스코어 = 각 감정에 등장하는 빈도 수 / 전체 빈도 수 
*sum = 1

형태소 분석기는 : 코모란 형태소 분석기를 사용함.
단어사전에는 : 명사 : 일반명사 , 고유명사 , 의존명사 , 형용사 , 동사 , 일반부사, 감탄사 , 외국어만 포함시킴

동사와 부사에는 “+다”를 붙여서 사전에 저장함. *명사와 동사/부사를 구분 짓기 위함 (ex 사과 , 사과하다 )

<img src = "https://user-images.githubusercontent.com/23625693/126860826-66afaafc-2ee7-46d0-9aeb-c573b7c92752.png" width="50%" height="50%">

감정 불용어 제거 ( ex : 있다 , 하다 , 나다 , 것 , 넌 , 다... )
*'있다'는 감정과 아무런 상관이 없지만 제거하지 않는 다면 감정 점수에 영향을 줌

### 형태소 분석기

Komoran 과 Okt 를 두고 고민을 많이 했는데 ( 둘다 속도가 비슷 )
정규화 측면에서 명사는 Okt 의 성능이 좋았고 , 동사는 코모란의 성능이 좋았다.
사실 두 형태소 분석기의 장/단점이 있지만
직관적으로 보았을 때 보기 편한 방법으로 코모란 형태소에 부사와 , 동사에 +'다'를 붙여주는 방식을 채택하였다.

코모란 형태소 분석기 결과
![image](https://user-images.githubusercontent.com/23625693/126860913-ca379fb4-a6ad-41cf-90b7-be93dde8f865.png)

## 하이브리드의 필요성
감정단어사전\-예시단어2개

<table style="border-collapse: collapse; width: 100%; height: 58px;" border="1" data-ke-align="alignLeft"><tbody><tr style="height: 18px;"><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="auto"><span></span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>love</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>fun</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>enthusiasm</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>happyness</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>sadness</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>anger</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>lonely</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>longing</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 18px;"><span><span style="color: #000000;" data-contrast="none"><span>fear</span></span><span style="color: #000000;">&nbsp;</span></span></td></tr><tr style="height: 20px;"><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>사랑</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.61</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="auto"><span></span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.01</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.02</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.18</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.07</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.04</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.06</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.02</span></span><span style="color: #000000;">&nbsp;</span></span></td></tr><tr style="height: 20px;"><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>마음</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.27</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.05</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.04</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.1</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.33</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="auto"><span></span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.09</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.07</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="height: 20px;"><span><span style="color: #000000;" data-contrast="none"><span>0.04</span></span><span style="color: #000000;">&nbsp;</span></span></td></tr></tbody></table>

"사랑" 이라는 단어는 사랑노래에 일반적으로(61%) 등장하지만 슬픈 노래에도(18%) 등장한다. 
이 사전을 가지고 점수화를 할 경우 사랑노래에 나오는 "사랑"이라는 단어에는 항상 슬픔이 18% 점수로 추가될 것이고 , 슬픈 노래에 나오는 "사랑"은 사랑감정이 61% 계속 점수에 추가될 것이다. 

그래서 중간에 긍/부정을 판단할 수 있는 딥러닝 모델(BERT)를 활용하여 문장에 대한 전체적인 맥락의 긍부정을 파악한다. 

그래서 긍정이 나올 경우

<table style="border-collapse: collapse; width: 100%;" border="1" data-ke-align="alignLeft"><tbody><tr><td style="width: 6.97674%;"><span><span style="color: #000000;" data-contrast="auto"><span></span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 7.55814%;"><span><span style="color: #000000;" data-contrast="none"><span>love</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 7.7907%;"><span><span style="color: #000000;" data-contrast="none"><span>fun</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 14.8837%;"><span><span style="color: #000000;" data-contrast="none"><span>enthusiasm</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 14.3023%;"><span><span style="color: #000000;" data-contrast="none"><span>happyness</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 11.6279%;"><span><span style="color: #000000;" data-contrast="none"><span>sadness</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 8.95349%;"><span><span style="color: #000000;" data-contrast="none"><span>anger</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 9.18605%;"><span><span style="color: #000000;" data-contrast="none"><span>lonely</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 10.814%;"><span><span style="color: #000000;" data-contrast="none"><span>longing</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 7.7907%;"><span><span style="color: #000000;" data-contrast="none"><span>fear</span></span><span style="color: #000000;">&nbsp;</span></span></td></tr><tr><td style="width: 6.97674%;"><span><span style="color: #000000;" data-contrast="none"><span>사랑</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 7.55814%;"><span><span style="color: #000000;" data-contrast="none"><span>0.61</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 7.7907%;"><span><span style="color: #000000;" data-contrast="auto"><span></span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 14.8837%;"><span><span style="color: #000000;" data-contrast="none"><span>0.01</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 14.3023%;"><span><span style="color: #000000;" data-contrast="none"><span>0.02</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 11.6279%;">&nbsp;</td><td style="width: 8.95349%;">&nbsp;</td><td style="width: 9.18605%;">&nbsp;</td><td style="width: 10.814%;">&nbsp;</td><td style="width: 7.7907%;">&nbsp;</td></tr><tr><td style="width: 6.97674%;"><span><span style="color: #000000;" data-contrast="none"><span>마음</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 7.55814%;"><span><span style="color: #000000;" data-contrast="none"><span>0.27</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 7.7907%;"><span><span style="color: #000000;" data-contrast="none"><span>0.05</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 14.8837%;"><span><span style="color: #000000;" data-contrast="none"><span>0.04</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 14.3023%;"><span><span style="color: #000000;" data-contrast="none"><span>0.1</span></span><span style="color: #000000;">&nbsp;</span></span></td><td style="width: 11.6279%;">&nbsp;</td><td style="width: 8.95349%;">&nbsp;</td><td style="width: 9.18605%;">&nbsp;</td><td style="width: 10.814%;">&nbsp;</td><td style="width: 7.7907%;">&nbsp;</td></tr></tbody></table>

뒤에 부정 감정들을 점수계산에서 제외시키고, 

부정이 나올 경우 

<table style="border-collapse: collapse; width: 100%;" border="1" data-ke-align="alignLeft"><tbody><tr><td><span><span style="color: #000000;" data-contrast="auto"><span></span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>love</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>fun</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>enthusiasm</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>happyness</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>sadness</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>anger</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>lonely</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>longing</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>fear</span></span><span style="color: #000000;">&nbsp;</span></span></td></tr><tr><td><span><span style="color: #000000;" data-contrast="none"><span>사랑</span></span><span style="color: #000000;">&nbsp;</span></span></td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><span><span style="color: #000000;" data-contrast="none"><span>0.18</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>0.07</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>0.04</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>0.06</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>0.02</span></span><span style="color: #000000;">&nbsp;</span></span></td></tr><tr><td><span><span style="color: #000000;" data-contrast="none"><span>마음</span></span><span style="color: #000000;">&nbsp;</span></span></td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><span><span style="color: #000000;" data-contrast="none"><span>0.33</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="auto"><span></span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>0.09</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>0.07</span></span><span style="color: #000000;">&nbsp;</span></span></td><td><span><span style="color: #000000;" data-contrast="none"><span>0.04</span></span><span style="color: #000000;">&nbsp;</span></span></td></tr></tbody></table>

앞에 있는 긍정단어들을 제외시킵니다. 

![image](https://user-images.githubusercontent.com/23625693/126861126-b238dc60-cf41-48fc-a8b3-0c7d0a374faa.png)
![image](https://user-images.githubusercontent.com/23625693/126861130-8fac497c-4621-4954-97e0-530d0318b896.png)
![image](https://user-images.githubusercontent.com/23625693/126861132-36b6981d-abac-49ec-9b0e-895669b91adc.png)
![image](https://user-images.githubusercontent.com/23625693/126861138-659b35f0-abf7-4b5b-9175-742fd4cb0bd9.png)

## 한국어 가사 감정 분석 흐름도

<img src = "https://user-images.githubusercontent.com/23625693/126861176-b57cc539-cb06-454d-9d2e-be4c2ee5480e.png" width="75%" height="75%">

## 분석 결과 

<img src = "https://user-images.githubusercontent.com/23625693/126861837-bc9984e0-5618-4271-8f0c-96ba55916670.png" width="75%" height="75%">

더 많은 결과 보기 : https://docs.google.com/spreadsheets/d/1hXIElY-9dCNQ2FIcEiRmiPLkta-r_2T0/edit#gid=1429848705
