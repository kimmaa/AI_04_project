import pandas as pd
import numpy as np
import sklearn
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


'''데이터 불러오기'''

df = pd.read_csv("D:\py4e\ds\project_01\imdb_crawl\imdb_crawltt10919420.tsv",delimiter='\t')
print(df.head())

print(df.columns)

# imdb데이터 컬럼 추가 "rate", "Date", "writer", "title", "review"
imdb_df = pd.read_csv(
    "D:\py4e\ds\project_01\imdb_crawl\imdb_crawltt10919420.tsv", delimiter='\t', names=["rate", "writer", "title", "review"])
print(imdb_df.head())
print(imdb_df.columns)

''' 데이터 전처리 '''
# 데이터 프레임 정보 확인
print(imdb_df.info())

# review에 내용이 없는 행 파악 - 모두 리뷰있음 확인 
print(len(imdb_df[imdb_df['review'] == '']))

print(len(imdb_df['review']))

# 학습용 데이터로 가공
print('평점 데이터 정보 :' ,imdb_df['rate'].describe())
print('평점 데이터 평점 :' ,imdb_df['rate'].unique())
print('평점 데이터 평점별 비율(오름차순) :\n' ,imdb_df['rate'].value_counts(ascending=True))


# 평점 8 이상 혹은 3 이하만 저장 (8 이상: 긍정적, 3 이하: 부정적)
'''
평점 데이터 평점별 비율(오름차순) :
0      17
3      43
2      47
4      53
5      84
1     112
6     130
7     187
8     246
9     345
10    402
'''
# re 라이브러리 import
import re
import string


# review 데이터 모두 소문자로 통일 
imdb_df['review'] = imdb_df['review'].str.lower()


# 불용어 제거
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer

# stopwords = [ 
#   "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because",
#   "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
#   "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
#   "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
#   "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
#   "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
#   "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
#   "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
#   "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
#   "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
  # "your", "yours", "yourself", "yourselves", "e", "t", "s", "i", "a", "n", "o", "r", "l"
  # ]
stop_words = set(stopwords.words('english'))


# 불용어 제거 함수
def remove_stopwords(data):
  data['review without stopwords'] = data['review'].apply(lambda x : ' '.join([word for word in x.split() if word not in stop_words]))
  return data



# 특수문자 제거 함수
def remove_tags(text):
    # result = re.sub('<.*?> ','',text)
    result = re.sub('\s+', ' ', text.replace("/[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]/gi", ""))
    return result

# 이모지 제거 함수
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


imdb_df_without_stopwords = remove_stopwords(imdb_df)
imdb_df_without_stopwords['clean_review']= imdb_df_without_stopwords['review without stopwords'].apply(lambda x : remove_tags(x))
imdb_df_without_stopwords['clean_review']= imdb_df_without_stopwords['clean_review'].apply(lambda x : deEmojify(x))
imdb_df_without_stopwords['clean_review'] = imdb_df_without_stopwords['clean_review'].str.replace('[{}]'.format(string.punctuation), ' ')

# print('2번째 리뷰',imdb_df_without_stopwords['clean_review'][1])
# print('11번째 리뷰', imdb_df_without_stopwords['clean_review'][10])

# print(imdb_df_without_stopwords.tail(10))


# 텍스트를 tokenize해서 adjective, verb, noun만 추출하는 함수
import nltk
from nltk.stem import PorterStemmer

'''
# 1) Tokenization 
# tokens = [word_tokenize(txt) for txt in texts] 
# # text1, text2, text3를 각각 소문자로 변환하고 
# word_tokenize()함수에 넣어 토큰화된 결과값을 리스트에 담는다. 
# # 2) stopword 제거 
# tokens_wo_stopword = [[word for word in token if not word in stop_words] for token in tokens] 
# # 3) stemming & 소문자 변환 
# tokens_stemmed = [[pst.stem(word) for word in token] for token in tokens_wo_stopword]

출처:  [Kelvin Overflow]
'''
def tokenize_text(text):
  #nltk에서 영어 불용어를 불러온다. 
  stop_words = set(stopwords.words('english')) 
  # 어간추출을 위한 인스턴스를 만든다. 
  pst = PorterStemmer() 
  # 1) Tokenization 
  tokens = [word_tokenize(txt) for txt in text] # text1, text2, text3를 각각 소문자로 변환하고 word_tokenize()함수에 넣어 토큰화된 결과값을 리스트에 담는다. 
  # 2) stopword 제거 
  tokens_wo_stopword = [[word for word in token if not word in stop_words] for token in tokens] 
  # 3) stemming & 소문자 변환 
  tokens_stemmed = [[pst.stem(word) for word in token] for token in tokens_wo_stopword]
  # 전처리 결과
  # print(tokens[0]) 
  # print(tokens_wo_stopword[0]) 
  return tokens_stemmed

#토큰화 확인
# print(tokenize_text(imdb_df_without_stopwords['clean_review'])[:2])

# 토큰화된 리뷰 데이터프레임에 넣기
imdb_df_without_stopwords['tokenize_review'] = tokenize_text(imdb_df_without_stopwords['clean_review'])
print(imdb_df_without_stopwords.head())

# 평점데이터 살펴보기 시각화
print('평점 데이터 평점별 비율(오름차순) :\n' ,imdb_df_without_stopwords['rate'].value_counts(ascending=True))
rate_rate = imdb_df_without_stopwords['rate'].value_counts(ascending=True)
sns.histplot(x=imdb_df_without_stopwords['rate'], kde=True,stat='count' , color='green' ,alpha = 0.5, bins=11)
# plt.show()



'''최빈단어 확인하기'''
from collections import Counter

# Counter 객체는 리스트요소의 값과 요소의 갯수를 카운트 하여 저장하고 있습니다.
# 카운터 객체는 .update 메소드로 계속 업데이트 가능합니다.
word_counts = Counter()

# 토큰화된 각 리뷰 리스트를 카운터 객체에 업데이트
imdb_df_without_stopwords['tokenize_review'].apply(lambda x: word_counts.update(x))

# 가장 많이 존재하는 단어 순으로 200개를 나열
# print(word_counts.most_common(20))
words_200 = word_counts.most_common(227)

# words_200의 불필요한 단어를 지운다
words_200.remove(('9', 113))
words_200.remove(('film', 99))
words_200.remove(('tv', 110))
words_200.remove(('mayb', 79))
words_200.remove(('7', 73))
words_200.remove(('episod', 723))
words_200.remove(('far', 84))
words_200.remove(('1', 131))
words_200.remove(('6', 130))
words_200.remove(('movi', 302))
words_200.remove(('squid', 330))
words_200.remove(('10', 207))
words_200.remove(('one', 636))
words_200.remove(('3', 90))
words_200.remove(('review', 96))
words_200.remove(('8', 68))
words_200.remove(('without', 72))
words_200.remove(('alreadi', 72))
words_200.remove(('seri', 987))
words_200.remove(('would', 403))
words_200.remove(('could', 246))
words_200.remove(('see', 240))
words_200.remove(('mani', 269))
words_200.remove(('2', 226)) 
words_200.remove(('seen', 192))
words_200.remove(('alic', 71))
words_200.remove(('ye', 67))
words_200.remove(('5', 65))
print(words_200)


# ''' 중간- 영화 전체리뷰의 word cloud 출력'''

wordcloud = WordCloud(background_color='white',width=800, height=600)
print(dict(words_200))


cloud = wordcloud.generate_from_frequencies(dict(words_200))
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(cloud, interpolation="lanczos")
plt.show()
cloud.to_file("./WC.png")

'''positive 7이상의 리뷰들만'''
imdb_squid = imdb_df_without_stopwords[['rate', 'tokenize_review']]
pos_squid = imdb_squid[imdb_squid.rate >= 7]
print(pos_squid)
pos_squid['tokenize_review'].apply(lambda x: word_counts.update(x))
pos_words_200 = word_counts.most_common(226)
pos_words_200.remove(('would', 670))
pos_words_200.remove(('seri', 1685))
pos_words_200.remove(('squid', 579))
pos_words_200.remove(('even', 520))
pos_words_200.remove(('also', 497))
pos_words_200.remove(('movi', 483))
pos_words_200.remove(('go', 461)) 
pos_words_200.remove(('thing', 451))
pos_words_200.remove(('2', 404)) 
pos_words_200.remove(('see', 401))
pos_words_200.remove(('could', 393))
pos_words_200.remove(('made', 363)) 
pos_words_200.remove(('10', 358))
pos_words_200.remove(('seen', 328))
pos_words_200.remove(('film', 161))
pos_words_200.remove(('saw', 160))
pos_words_200.remove(('review', 159))
pos_words_200.remove(('3', 143))
pos_words_200.remove(('far', 139))
pos_words_200.remove(('mayb', 131))
pos_words_200.remove(('7', 124))
pos_words_200.remove(('alic', 116))
pos_words_200.remove(('episod', 1193))
pos_words_200.remove(('one', 1088))
pos_words_200.remove(('alreadi', 111))
pos_words_200.remove(('8', 109))
print(pos_words_200)

# 7이상의 리뷰 워드클라우드
pos_wordcloud = WordCloud(background_color='white',width=800, height=600)
print(dict(pos_words_200))

pos_cloud = pos_wordcloud.generate_from_frequencies(dict(pos_words_200))
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(pos_cloud, interpolation="lanczos")
plt.show()
pos_cloud.to_file("./positive_WC.png")

'''Negative 3이하의 리뷰들만'''
neg_squid = imdb_squid[imdb_squid.rate <= 3]
print(neg_squid)
neg_squid['tokenize_review'].apply(lambda x: word_counts.update(x))
neg_squid_200 = word_counts.most_common(231)
print(neg_squid_200)
neg_squid_200.remove(('seri', 1799))
neg_squid_200.remove(('episod', 1271))
neg_squid_200.remove(('one', 1150))
neg_squid_200.remove(('squid', 605))
neg_squid_200.remove(('would', 729))
neg_squid_200.remove(('movi', 528))
neg_squid_200.remove(('also', 529))
neg_squid_200.remove(('season', 575))
neg_squid_200.remove(('mani', 472))
neg_squid_200.remove(('could', 415))
neg_squid_200.remove(('see', 428))
neg_squid_200.remove(('2', 419))
neg_squid_200.remove(('10', 382))
neg_squid_200.remove(('seen', 349))
neg_squid_200.remove(('1', 243))
neg_squid_200.remove(('6', 220))
neg_squid_200.remove(('seem', 208))
neg_squid_200.remove(('9', 202))
neg_squid_200.remove(('saw', 177))
neg_squid_200.remove(('review', 177))
neg_squid_200.remove(('film', 178))
neg_squid_200.remove(('howev', 197))
neg_squid_200.remove(('3', 151))
neg_squid_200.remove(('far', 148))
neg_squid_200.remove(('mayb', 141))
neg_squid_200.remove(('alic', 125))
neg_squid_200.remove(('without', 123))
neg_squid_200.remove(('ye', 124))
neg_squid_200.remove(('7', 128))
neg_squid_200.remove(('alreadi', 123))
neg_squid_200.remove(('8', 114))
print(neg_squid_200)

# 평점 3이하의 리뷰 워드클라우드
neg_wordcloud = WordCloud(background_color='white',width=800, height=600)
print(dict(neg_squid_200))

neg_cloud = neg_wordcloud.generate_from_frequencies(dict(neg_squid_200))
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(neg_cloud, interpolation="lanczos")
plt.show()
neg_cloud.to_file("./negative_WC.png")


# 단어 평균값 구하기 

# 단어 수
imdb_df_without_stopwords['num_words'] = imdb_df_without_stopwords['tokenize_review'].apply(lambda x: len(str(x).split()))
# 중복을 제거한 단어 수
imdb_df_without_stopwords['num_uniq_words'] = imdb_df_without_stopwords['tokenize_review'].apply(lambda x: len(set(str(x).split())))

# 첫 번째 리뷰에 
x = imdb_df_without_stopwords['tokenize_review'][0]
x = str(x).split()
print(len(x))
print(x[:10])

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(18, 6)
print('Avg. value of words per review :', imdb_df_without_stopwords['num_words'].mean())
print('Median value of words per review : ', imdb_df_without_stopwords['num_words'].median())
sns.distplot(imdb_df_without_stopwords['num_words'], bins=100, ax=axes[0])
axes[0].axvline(imdb_df_without_stopwords['num_words'].median(), linestyle='dashed')
axes[0].set_title('Distribution of word count by reviews')

print('Avg. value of unique words per review :', imdb_df_without_stopwords['num_uniq_words'].mean())
print('Median value of Unique words per review : ', imdb_df_without_stopwords['num_uniq_words'].median())
sns.distplot(imdb_df_without_stopwords['num_uniq_words'], bins=100, color='g', ax=axes[1])
axes[1].axvline(imdb_df_without_stopwords['num_uniq_words'].median(), linestyle='dashed')
axes[1].set_title('Distribution of number of unique words by review')
# plt.show()
'''
리뷰 별 단어 평균값 : 44.63805522208884
리뷰 별 단어 중간값 27.0
리뷰 별 고유 단어 평균값 : 37.48979591836735
리뷰 별 고유 단어 중간값 25.0
'''

# 사이킷런의 CountVectorizer를 통해 피처 생성-----------여기부터 진행 필요요
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# 튜토리얼과 다르게 파라메터 값을 수정
# 파라메터 값만 수정해도 캐글 스코어 차이가 크게 남
vectorizer = CountVectorizer(analyzer = 'word', tokenizer = None, preprocessor = None, \
  stop_words = None, min_df = 2, # 토큰이 나타날 최소 문서 개수 
  ngram_range=(1, 3), max_features = 20000)
vectorizer

# 속도 개선을 위해 파이프라인을 사용하도록 개선
# 참고 : https://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph
pipeline = Pipeline([
    ('vect', vectorizer),
])  

train_data_features = pipeline.fit_transform(imdb_df_without_stopwords)
train_data_features
train_data_features.shape
vocab = vectorizer.get_feature_names()
print(vocab[:10])

# X_texts = []
# y = []

# for rate, comment in zip(imdb_df_without_stopwords['rate'], imdb_df_without_stopwords['tokenize_review']):
#   if 4 <= rate < 7: 
#     continue  
#     # 평점이 4~7인 영화는 애매하기 때문에 학습데이터로 사용하지 않음

#   # 위에서 만들었던 함수로 comment 쪼개기
#   X_texts.append(imdb_df_without_stopwords['tokenize_review'])

#   y.append(1 if rate >= 7 else -1)
#     # 평점이 8 이상이면(8,9,10) 값을 1로 지정 (positive)
#     # 평점이 3 이하이면(1,2,3) 값을 0로 지정 (negative)

# print(f'원래 text 수: {len(imdb_df_without_stopwords)}')
# print(f'평점 3 이하 혹은 8 이상인 text 수: {len(X_texts)}')    
# print(X_texts[:5])

# # train_test_split
# from sklearn.feature_extraction.text import CountVectorizer   # tf-idf 방식을 사용하려면 대신 TfidfVectorizer를 import
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# X_train_texts, X_test_texts, y_train, y_test = train_test_split(X_texts, y, test_size=0.2, random_state=0)
# # CountVectorizer로 vector화
# tf_vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
# X_train_tf = tf_vectorizer.fit_transform(X_train_texts)  # training data에 맞게 fit & training data를 transform
# X_test_tf = tf_vectorizer.transform(X_test_texts) # test data를 transform

# vocablist = [word for word, number in sorted(tf_vectorizer.vocabulary_.items(), key=lambda x:x[1])]  # 단어들을 번호 기준 내림차순으로 저장

# ## 확인해보기
# print(X_train_tf[:1], '\n')
# print(X_test_tf[:1], '\n')
# print(vocablist[:3])