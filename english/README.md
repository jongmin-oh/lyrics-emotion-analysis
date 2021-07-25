# 영어 가사 감정 분류

## 데이터
한국어 처럼 영어도 마찬가지로 가사도메인에 대한 데이터셋을 구축하여 분석을 진행하는 것이 가장 좋다고 생각하지만 <br>
영어에 대한 라벨링을 진행하는 것이 시간 소요가 많고 정확도가 많이 떨어 질 수 있다고 생각하여<br>
영어에 대한 감정을 나타내는 다른 TEXT 데이터 셋를 활용하여 감정 분류를 진행 ( *자세한 내용 전 페이지 참고 )

1. 평서문 감정 데이터 셋
2. 트위터 감정 데이터 셋

또 데이터 셋의 불균형 문제가 심각했지만<br>
두 가지 데이터 셋을 적절하게 조합하고 빈도 수를 고려하여 불균형을 완화했습니다.

<table style="border-collapse: collapse; width: 50.3488%;" border="1" data-ke-align="alignLeft"><tbody><tr><td style="width: 13.1396%;">&nbsp;</td><td style="width: 12.6744%;"><span><span>Tweet</span></span></td><td style="width: 13.7209%;"><span><span>Natural</span></span></td><td style="width: 10.6977%;"><span><span>Total</span></span></td></tr><tr><td style="width: 13.1396%;"><span><span>suprise</span></span></td><td style="width: 12.6744%;"><span><span>2000</span></span></td><td style="width: 13.7209%;"><span><span>1000</span></span></td><td style="width: 10.6977%;"><span><span>3000</span></span></td></tr><tr><td style="width: 13.1396%;"><span><span>Love</span></span></td><td style="width: 12.6744%;"><span><span>3000</span></span></td><td style="width: 13.7209%;"><span><span>2000</span></span></td><td style="width: 10.6977%;"><span><span>5000</span></span></td></tr><tr><td style="width: 13.1396%;"><span><span>Happy</span></span></td><td style="width: 12.6744%;"><span><span>3000</span></span></td><td style="width: 13.7209%;"><span><span>3000</span></span></td><td style="width: 10.6977%;"><span><span>6000</span></span></td></tr><tr><td style="width: 13.1396%;"><span><span>Sadness</span></span></td><td style="width: 12.6744%;"><span><span>3000</span></span></td><td style="width: 13.7209%;"><span><span>3000</span></span></td><td style="width: 10.6977%;"><span><span>6000</span></span></td></tr><tr><td style="width: 13.1396%;"><span><span>Anger</span></span></td><td style="width: 12.6744%;"><span><span>2000</span></span></td><td style="width: 13.7209%;"><span><span>3000</span></span></td><td style="width: 10.6977%;"><span><span>5000</span></span></td></tr><tr><td style="width: 13.1396%;"><span><span>Fear</span></span></td><td style="width: 12.6744%;"><span><span>1000</span></span></td><td style="width: 13.7209%;"><span><span>3000</span></span></td><td style="width: 10.6977%;"><span><span>4000</span></span></td></tr></tbody></table>

## 전처리

가사 데이터는 음원 홈페이지에서 크롤링한 것이기 때문에 오탈자와 전처리를 작업을 형태소 분석을 통한 정규화 작업이 전부였지만 <br>
트위터 메신저 데이터의 경우 오탈자와 줄임말이 많기 때문에 다양한 전처리 작업을 거쳤습니다.

### text_hammer 패키지 사용 (* 영어 전처리를 도와주는 패키지 )

 1. 문자변환 , 소문자 변환 ( str 형태로 바꿔주고 , 모든 알파벳을 소문자로 변환 )
 2. 줄임표 정규화 ( you're -> you are , i'm -> i am )
 4. 이메일 형식 제거
 5. HTML 태그 제거
 6. 불용어(stopwords)제거
 7. 특수문자 제거
 8. 단어 원형 변경(WordNetLemmatizer)

<table style="border-collapse: collapse; width: 32.6744%;" border="1" data-ke-align="alignLeft" data-ke-style="style12"><tbody><tr><td style="width: 15.814%;"><b><span><span>Word</span></span></b></td><td style="width: 16.7442%;"><b><span><span>Lemmatized</span></span></b></td></tr><tr><td style="width: 15.814%;"><b><span><span>Dies</span></span></b></td><td style="width: 16.7442%;"><b><span><span>Die</span></span></b></td></tr><tr><td style="width: 15.814%;"><b><span><span>Watched</span></span></b></td><td style="width: 16.7442%;"><b><span><span>Watch</span></span></b></td></tr><tr><td style="width: 15.814%;"><b><span><span>Has</span></span></b></td><td style="width: 16.7442%;"><b><span><span>Have</span></span></b></td></tr></tbody></table>

<table style="border-collapse: collapse; width: 100%;" border="1" data-ke-align="alignLeft"><tbody><tr><td style="width: 100%;">처리&nbsp;전&nbsp;:<br>@tiffanylue&nbsp;i&nbsp;know&nbsp;i&nbsp;was&nbsp;listenin&nbsp;to&nbsp;bad&nbsp;habit&nbsp;earlier&nbsp;and&nbsp;i&nbsp;started&nbsp;freakin&nbsp;at&nbsp;his</td></tr><tr><td style="width: 100%;">처리&nbsp;후:<br>I&nbsp;Know&nbsp;I&nbsp;be&nbsp;listenin&nbsp;bad&nbsp;habit&nbsp;earlier&nbsp;I&nbsp;start&nbsp;freakin</td></tr></tbody></table>

## 모델링

영어의 경우 한국어와 같이 하이브리드 방법론을 사용했지만 가사 도메인이 아닌 데이터 셋에 대해서 생성된 감정 사전(lexicon)에 여럿 단어가 부정확하게 나타났습니다.<br>
그래서 BERT 다중 분류 모델을 사용했고 BERT 의 경우 한국어 다중 분류를 할때 성능이 엄청 떨어졌지만 영어에서는 높은 성능을 나타내었습니다.

### Bert Base 모델 Distilbert 사용<br>
*BERT를 40%로 줄이고 60% 빠르게 연산하면서 97%의 성능을 유지함.

학습데이터 : 20300<br>
검증데이터 : 8700<br>

MAX_LEN = 70<br>

Input : input_ids , input_mask <br>

Layers :<br>
GlobalMaxPool1Dd( )(embeddings) <br>
Dense( 128 , activation = 'relu') <br>
Dropout(0.1)<br>
Dense(32, activation = 'relu )<br>

#output<br>
Dense(6, activation = 'sigmoid’ )<br>

Compile :<br>
Optimizer : Adam<br>
Loss : CategoricalCrossentropy Metric : categoricalAccuracy<br>

Fit :<br>
Epoch : 50 Batch_size = 64<br>

![image](https://user-images.githubusercontent.com/23625693/126885838-7ed3455b-68c4-43b4-bab5-7fb73d3355f5.png)


### Classification Report

#### Accuracy : 81%

<table style="border-collapse: collapse; width: 58.8372%; height: 124px;" border="1" data-ke-align="alignLeft"><tbody><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">&nbsp;</td><td style="width: 11.1628%; height: 18px;">Precision</td><td style="width: 12.5581%; height: 18px;">Recall</td><td style="width: 12.907%; height: 18px;">F1-score</td><td style="width: 11.0465%; height: 18px;">Support</td></tr><tr style="height: 16px;"><td style="width: 11.1628%; height: 16px;">surprise</td><td style="width: 11.1628%; height: 16px;">0.70</td><td style="width: 12.5581%; height: 16px;">0.71</td><td style="width: 12.907%; height: 16px;">0.70</td><td style="width: 11.0465%; height: 16px;">900</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">love</td><td style="width: 11.1628%; height: 18px;">0.79</td><td style="width: 12.5581%; height: 18px;">0.81</td><td style="width: 12.907%; height: 18px;">0.80</td><td style="width: 11.0465%; height: 18px;">1500</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">joy</td><td style="width: 11.1628%; height: 18px;">0.81</td><td style="width: 12.5581%; height: 18px;">0.76</td><td style="width: 12.907%; height: 18px;">0.78</td><td style="width: 11.0465%; height: 18px;">1800</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">sadness</td><td style="width: 11.1628%; height: 18px;">0.81</td><td style="width: 12.5581%; height: 18px;">0.81</td><td style="width: 12.907%; height: 18px;">0.81</td><td style="width: 11.0465%; height: 18px;">1800</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">anger</td><td style="width: 11.1628%; height: 18px;">0.86</td><td style="width: 12.5581%; height: 18px;">0.90</td><td style="width: 12.907%; height: 18px;">0.88</td><td style="width: 11.0465%; height: 18px;">1500</td></tr><tr style="height: 18px;"><td style="width: 11.1628%; height: 18px;">fear</td><td style="width: 11.1628%; height: 18px;">0.88</td><td style="width: 12.5581%; height: 18px;">0.88</td><td style="width: 12.907%; height: 18px;">0.88</td><td style="width: 11.0465%; height: 18px;">1200</td></tr></tbody></table>

## 테스트 결과
