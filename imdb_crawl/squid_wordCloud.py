import pandas as pd
import numpy as np
import re

df = pd.read_csv('D:\py4e\ds\project_01\imdb_crawl\MyCloud.csv', delimiter='\t')
# print(df)
print(df['tokenize_review'].head())

# tokenize_review를 리스트에 담는당
print(len(df['tokenize_review']))

df_sel = df[['rate', 'tokenize_review']]
print(df_sel)

from collections import Counter

# Counter 객체는 리스트요소의 값과 요소의 갯수를 카운트 하여 저장하고 있습니다.
# 카운터 객체는 .update 메소드로 계속 업데이트 가능합니다.
word_counts = Counter()

# 토큰화된 각 리뷰 리스트를 카운터 객체에 업데이트
df_sel['tokenize_review'].apply(lambda x: word_counts.update(x))

# 가장 많이 존재하는 단어 순으로 10개를 나열
print(word_counts.most_common(20))

# ##groupby()함수의 by옵션에 칼럼을 입력하면 대상 칼럼으로 그룹핑됩니다.
# imdb_groupby = df_sel.groupby(by='rate')
# print(imdb_groupby.count())

# imdb_groupby.to_csv('./test_cl.csv', header=True, sep='\t')

# reviews_list = []
# for i in range(len(df['tokenize_review'])):
#     reviews_list.append(df['tokenize_review'][i])


# # res = [x for x in reviews_list]
# # # print(res[:2])
# print(reviews_list[:2])

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# myWC = WordCloud(background_color="white").generate(df['tokenize_review'])
# plt.figure(figsize=(5,5))

# plt.imshow(myWC, interpolation="lanczos")

# plt.axis('off')

# plt.show()