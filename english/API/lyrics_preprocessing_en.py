import text_hammer as th
import re

def get_upper_idx(lyrics):
    """
    문장의 맨앞에 있는 글자가 소문자인지 체크하고 대문자가 나오면
    대문자의 idx를 저장함.
    """
    upper_idx = []
    idx = 0
    for sentence in lyrics:
        # 특수문자 제거
        sentence = th.remove_special_chars(sentence)
        try :
            if sentence[0].isupper():
                upper_idx.append(idx)
            idx = idx + 1
        except :
            pass
    return upper_idx

def join_sentence(index_list , corpus):
    """
    대문자가 등장한 경우만 문장의 시작이라고 판단하여 문장을 재구성함.
    DB에서 데이터를 가져올때 개행문자로 쪼개면 너무 잘게 쪼개져서
    문장의 문맥을 제대로 파악할 수 없는 문제를 개선하기 위함.
    """
    result = []
    for i in range(len(index_list)):
        try:
            result.append(" ".join(corpus[index_list[i]:index_list[i+1]]))
        except IndexError:
            result.append(corpus[index_list[-1] -1])
    return result

# 가사 문장별 분리
def lyrics_to_corpus(lyrics):
    lyrics = re.split('\r\n|\n',lyrics)
    corpus = join_sentence(get_upper_idx(lyrics),lyrics)
    #중복제거
    corpus = list(set(corpus))
    return corpus

# 가사 문장별 전처리
def sentence_preprocessing(corpus):
    # 이 부분 re.로 처리하는 방법
    #앞뒤 공백 제거 & 소문자
    corpus = [sentence.strip().lower() for sentence in corpus]
    # 이메일 형식 제거
    corpus = [th.remove_emails(sentence) for sentence in corpus]
    # html 태그 제거
    corpus = [th.remove_html_tags(sentence) for sentence in corpus]
    # 특수문자 제거
    corpus = [th.remove_special_chars(sentence) for sentence in corpus]
    return corpus