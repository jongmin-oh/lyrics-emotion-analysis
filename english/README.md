# 영어 가사 감정 분류

## 데이터
한국어 처럼 영어도 마찬가지로 가사도메인에 대한 데이터셋을 구축하여 분석을 진행하는 것이 가장 좋다고 생각하지만 <br>
영어에 대한 라벨링을 진행하는 것이 시간 소요가 많고 정확도가 많이 떨어 질 수 있다고 생각하여<br>
영어에 대한 감정을 나타내는 다른 TEXT 데이터 셋를 활용하여 감정 분류를 진행 ( *자세한 내용 전 페이지 참고 )

1. 평서문 감정 데이터 셋
2. 트위터 감정 데이터 셋

또 데이터 셋의 불균형 문제가 심각했지만<br>
두 가지 데이터 셋을 적절하게 조합하고 빈도 수를 고려하여 불균형을 완화했습니다.

## 전처리

가사 데이터는 음원 홈페이지에서 크롤링한 것이기 때문에 오탈자와 전처리를 작업을 형태소 분석을 통한 정규화 작업이 전부였지만 <br>
트위터 메신저 데이터의 경우 오탈자와 줄임말이 많기 때문에 다양한 전처리 작업을 거쳤습니다.

### text_hammer 패키지 사용 (* 영어 전처리를 도와주는 패키지 )

 1. 문자변환 , 소문자 변환 ( str 형태로 바꿔주고 , 모든 알파벳을 소문자로 변환 )
 2. 줄임표 정규화 ( you're -> you are , i'm -> i am )
 3. 이메일 형식 제거
 4. HTML 태그 제거
 5. 불용어(stopwords)제거
 6. 특수문자 제거
 7. 단어 원형 변경 ( ran -> run , went -> go )

## 모델링

영어의 경우 한국어와 같이 하이브리드 방법론을 사용했지만 가사 도메인이 아닌 데이터 셋에 대해서 생성된 감정 사전(lexicon)에 여럿 단어가 부정확하게 나타났습니다.<br>
그래서 BERT 다중 분류 모델을 사용했고 BERT 의 경우 한국어 다중 분류를 할때 성능이 엄청 떨어졌지만 영어에서는 높은 성능을 나타내었습니다.

Bert Base 모델 Distilbert 사용
*BERT를 40%로 줄이고 60% 빠르게 연산하면서 97%의 성능을 유지함.

학습데이터 : 14000 
검증데이터 : 7000

MAX_LEN = 70

Input : input_ids , input_mask 

Layers :
GlobalMaxPool1Dd( )(embeddings) 
Dense( 128 , activation = 'relu') 
Dropout(0.1)
Dense(32, activation = 'relu )

#output
Dense(6, activation = 'sigmoid’ )

Compile :
Optimizer : Adam
Loss : CategoricalCrossentropy Metric : categoricalAccuracy

Fit :
Epoch : 1 Batch_size = 36

정제가 잘 되어있는 데이터 셋이고 평서문데이터는 특정한 규칙이 존재하기 때문에 학습이 잘 된듯합니다.

#### Classification Report

<table style="border-collapse: collapse; width: 33.3721%; height: 36px;" border="1" data-ke-align="alignLeft"><tbody><tr style="height: 18px;"><td style="width: 50%; height: 18px;">Accuracy</td><td style="width: 50%; height: 18px;">0.9259</td></tr><tr style="height: 18px;"><td style="width: 50%; height: 18px;">F1 score</td><td style="width: 50%; height: 18px;">0.8842</td></tr></tbody></table>

<table style="border-collapse: collapse; width: 58.8372%; height: 126px;" border="1" data-ke-align="alignLeft"><tbody><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">&nbsp;</td><td style="width: 11.1628%; height: 18px;">Precision</td><td style="width: 12.5581%; height: 18px;">Recall</td><td style="width: 12.907%; height: 18px;">F1-score</td><td style="width: 11.0465%; height: 18px;">Support</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">surprise</td><td style="width: 11.1628%; height: 18px;">0.92</td><td style="width: 12.5581%; height: 18px;">0.97</td><td style="width: 12.907%; height: 18px;">0.94</td><td style="width: 11.0465%; height: 18px;">664</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">love</td><td style="width: 11.1628%; height: 18px;">0.97</td><td style="width: 12.5581%; height: 18px;">0.96</td><td style="width: 12.907%; height: 18px;">0.97</td><td style="width: 11.0465%; height: 18px;">586</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">joy</td><td style="width: 11.1628%; height: 18px;">0.90</td><td style="width: 12.5581%; height: 18px;">0.94</td><td style="width: 12.907%; height: 18px;">0.92</td><td style="width: 11.0465%; height: 18px;">263</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">sadness</td><td style="width: 11.1628%; height: 18px;">0.94</td><td style="width: 12.5581%; height: 18px;">0.85</td><td style="width: 12.907%; height: 18px;">0.89</td><td style="width: 11.0465%; height: 18px;">247</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">anger</td><td style="width: 11.1628%; height: 18px;">0.90</td><td style="width: 12.5581%; height: 18px;">0.77</td><td style="width: 12.907%; height: 18px;">0.83</td><td style="width: 11.0465%; height: 18px;">185</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">fear</td><td style="width: 11.1628%; height: 18px;">0.68</td><td style="width: 12.5581%; height: 18px;">0.83</td><td style="width: 12.907%; height: 18px;">0.75</td><td style="width: 11.0465%; height: 18px;">54</td></tr></tbody></table>
