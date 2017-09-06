import pandas as pd

from nltk.corpus import stopwords
import nltk
import time
from collections import Counter
from nltk import pos_tag
from nltk import word_tokenize
from tqdm import tqdm
import sys



def text_processing(df):
	print('start text processing')
	start=time.clock()
	df['text']=[' '.join(i.split()) for i in tqdm(df.text)]
	print('remove \\n, \\t, etc.')
	print('using time:', float(time.clock()-start))
	df['text']=[i.lower() for i in df.text]
	print('lower all words')
	print('using time:', float(time.clock()-start))
	df['filtered_text']=[' '.join([i for i in j.split() if i not in stopwords.words('english') ]) for j in tqdm(df.text)]
	print('remove stopwords')
	print('using time:', float(time.clock()-start))
	df['lemmatize_text']=[' '.join([lem.lemmatize(i) for i in j.split() ]) for j in tqdm(df.text)]
	print('Lemmatisation text with stopwords')
	print('using time:', float(time.clock()-start))
	df['lemmatize_filtered_text']=[' '.join([lem.lemmatize(i) for i in j.split() ]) for j in tqdm(df.filtered_text)]
	print('Lemmatisation text without stopwords')
	print('using time:', float(time.clock()-start))
	print('Finish')

if __name__ == '__main__':
	path=sys.argv
	df=pd.read_pickle(path[1])
	lem=nltk.wordnet.WordNetLemmatizer()
	text_processing(df)
	df.to_pickle(path[2])