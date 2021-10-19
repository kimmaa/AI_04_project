import pandas as pd
import numpy as np

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
stopwords = [ 
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because",
  "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
  "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
  "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
  "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
  "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
  "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
  "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
  "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
  "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
  "your", "yours", "yourself", "yourselves" 
  ]

# 불용어 제거 함수
def remove_stopwords(data):
  data['review without stopwords'] = data['review'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stopwords)]))
  return data

# 특수문자 제거 함수
def remove_tags(string):
    result = re.sub('<.*?> ','',string)
    return result

imdb_df_without_stopwords = remove_stopwords(imdb_df)
imdb_df_without_stopwords['clean_review']= imdb_df_without_stopwords['review without stopwords'].apply(lambda cw : remove_tags(cw))
imdb_df_without_stopwords['clean_review'] = imdb_df_without_stopwords['clean_review'].str.replace('[{}]'.format(string.punctuation), ' ')
# imdb_df_without_stopwords['clean_review']= remove_tags(string)


print('2번째 리뷰',imdb_df_without_stopwords['clean_review'][1])
print('11번째 리뷰', imdb_df_without_stopwords['clean_review'][10])

print(imdb_df_without_stopwords.tail(3))

from collections import Counter

# Counter 객체는 리스트요소의 값과 요소의 갯수를 카운트 하여 저장하고 있습니다.
# 카운터 객체는 .update 메소드로 계속 업데이트 가능합니다.
word_counts = Counter()

# 토큰화된 각 리뷰 리스트를 카운터 객체에 업데이트 합니다. 
print(remove_stopwords(imdb_df_without_stopwords))

imdb_df_without_stopwords['clean_review'].apply(lambda x: word_counts.update(x))
imdb_df_without_stopwords['review without stopwords'].apply(lambda x: word_counts.update(x))

# 가장 많이 존재하는 단어 순으로 10개를 나열
print(word_counts.most_common(10))
'''
[(' ', 178428), ('e', 117048), ('t', 70648), ('s', 67204), ('i', 67050), ('a', 66082), ('n', 63984), ('o', 61194), ('r', 53554), ('l', 48040)]
미쵸따...우짜지.... 
'''

# clean review를 리스트에 담는당
print(len(imdb_df_without_stopwords['clean_review']))

reviews_list = []
for i in range(len(imdb_df_without_stopwords['clean_review'])):
  reviews_list.append(imdb_df_without_stopwords['clean_review'][i])

sentiment = imdb_df_without_stopwords['sentiment'] 

# print(reviews_list) # 이모티콘들 갑툭튀...미치게땅......