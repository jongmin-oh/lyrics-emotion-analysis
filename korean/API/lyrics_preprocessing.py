import re

# 가사 문장별 분리
def lyrics_to_corpus(lyrics):
    lyrics = re.split('\r\n|\n',lyrics)
    #중복제거
    corpus = list(set(lyrics))
    return corpus

# 가사 문장별 전처리
def sentence_preprocessing(corpus):
    #특수문자 제거
    corpus = [re.sub('[^\w]' ," ", sentence) for sentence in corpus]
    #앞뒤 공백 제거
    corpus = [sentence.strip() for sentence in corpus]
    #길이 5개 미만 제거
    corpus = [sentence for sentence in corpus if len(sentence) > 5]
    #같은 단어 반복 제거 ex) Oh Oh Oh Oh
    corpus = [list(set(sentence.split())) for sentence in corpus]
    # 2차원 리스트 -> 1차원
    corpus = [" ".join(sentence) for sentence in corpus]
    return corpus


#통일 시켜주기 위함
def lexicon_tokenizer_komoran(corpus,kmoran):
    temp = []
    for i in kmoran.pos(corpus):
        if (i[1] == 'NNG' or i[1] == 'NNP'):
            temp.append(i[0])
        elif (i[1] == 'VV' or i[1] == 'VA'):
            temp.append(i[0] + "다")
        elif (i[1] == 'SL'):
            temp.append(i[0].lower())
    return temp

# def sentence_preprocessing(corpus):
#     #특수문자 제거
#     corpus = [re.sub('[^\w]' ," ", sentence) for sentence in corpus]
#     corpus = [re.sub('[^가-힣+]' ," ", sentence) for sentence in corpus]
#     #앞뒤 공백 제거
#     corpus = [sentence.strip() for sentence in corpus]
#     return corpus

#한글 영어 구분 : 한글이 하나라도 있으면 한글 분류
def EnglishOrKorean(input_s):
    k_count = 0
    e_count = 0
    for c in input_s:
        if ord('가') <= ord(c) <= ord('힣'):
            k_count+=1
        elif ord('a') <= ord(c.lower()) <= ord('z'):
            e_count+=1
    return "k" if k_count>1 else "e"

# 한글 영어 구분 : 퍼센트 입력
def isKorean_percent(input_s , percentage):
    k_count = 0
    e_count = 0
    for c in input_s:
        if ord('가') <= ord(c) <= ord('힣'):
            k_count+=1
        elif ord('a') <= ord(c.lower()) <= ord('z'):
            e_count+=1
    if k_count ==0:
        return 0
    else :
        percent = k_count / (k_count + e_count)
        return 1 if percent>percentage else 0

def isRemoveData(title):
    # 타이틀에 붙어있는 MR , Inst. 필터
    if 'inst.' in title or 'Inst.' in title:
        return 1
    if 'Instrumental' in title or 'instrumental' in title:
        return 1
    elif 'MR' in title:
        return 1
    else :
        return 0