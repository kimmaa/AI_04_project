import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer

text1 = "I Love You." 
text2 = "I can't stop!"
text3 = "A processing interface for removing morphological affixes from words. This process is known as stemming."

# 데이터셋에서 X에 해당합니다.
texts = [text1, text2, text3]

# nltk에서 영어 불용어를 불러온다.
stop_words = set(stopwords.words('english')) 
# 어간추출을 위한 인스턴스를 만든다.
pst = PorterStemmer()

# 토큰화 및 불용어 제거
tokens = []
for txt in texts:
    token = word_tokenize(txt) # i love korea -> 'i', 'love', 'korea'로 나누어 준다.
    non_stopwords = [pst.stem(t) for t in token if not t in stop_words] # 불용어가 아닌 token을 어간추출해서 리스트에 넣는다.
    tokens.append(non_stopwords)

# 전처리 결과
print(tokens)

