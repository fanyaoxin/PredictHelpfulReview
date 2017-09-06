## the traditional features include: 
#'review_all_length', 'review_exceptstop_length', 'noun', 'adj', 'adv', 'verb', 'sentiment_score'

import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from nltk import pos_tag



def construct_other_text_features(df):
	df['review_all_length']=[len(i.split(' ')) for i in tqdm(df.text)]
	df['review_exceptstop_length']=[len(i.split(' ')) for i in tqdm(df.filtered_text)]
def compute_tag(text):
	l=Counter([j for i,j in pos_tag(word_tokenize(text))])
	num=sum([l[i] for i in ['NN','NNP','NNS','NNPS']])
	adj=sum([l[i] for i in ['JJ','JJR','JRS']])
	adv=sum([l[i] for i in ['RB','RBR','RBS']])
	verb=sum([l[i] for i in ['VB','VBD','VBG','VBN','VBP','VBZ']])
	return list([num,adj,adv,verb])
def construct_words_tag_feature(df):
	df['noun']=[compute_tag(i)[0] for i in tqdm(df['lemmatize_filtered_text'])]
	df['adj']=[compute_tag(i)[1] for i in tqdm(df['lemmatize_filtered_text'])]
	df['adv']=[compute_tag(i)[2] for i in tqdm(df['lemmatize_filtered_text'])]
	df['verb']=[compute_tag(i)[3] for i in tqdm(df['lemmatize_filtered_text'])]
def construct_sentiment_score(df):
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score']=[sid.polarity_scores(i)['compound'] for i in tqdm(df.filtered_text)]


if __name__ == '__main__':
	path=sys.argv
	df=pd.read_pickle(path[1])
	construct_sentiment_score(df)
	construct_words_tag_feature(df)
	construct_other_text_features(df)
	df.to_pickle(path[2])

