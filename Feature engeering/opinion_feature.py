## take all reivews into computation (find all aspect)
## pick the most 200 frequent aspects
## divided these 200 aspects into food, service, restaurant, others, price

from opinion_mining.score_aspect import SentimentScorer, get_sentences_by_aspect
from opinion_mining.extract_aspects import get_sentences, tokenize, pos_tag, aspects_from_tagged_sents
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np

sentiment_scorer = SentimentScorer()
def compute_score(reviews,aspects):
	aspect_sentences = get_sentences_by_aspect(aspects, reviews)
	scores = [sentiment_scorer.score(sent) for sent in aspect_sentences]
	return sum(scores)

def mean_compute_score(reviews,aspects):
	aspect_sentences = get_sentences_by_aspect(aspects, reviews)
	scores = [sentiment_scorer.score(sent) for sent in aspect_sentences]
	if len(scores)==0:
		return 0
	return np.mean(scores)

def construct_review_aspect_score(df):
	df["environment_score"]=[compute_score(i, environments) for i in tqdm(df.text)]
	df["mean_environment_score"]=[mean_compute_score(i, environments) for i in tqdm(df.text)]
	print('finish one')
	df["other_score"]=[compute_score(i, others) for i in tqdm(df.text)]
	df["mean_other_score"]=[mean_compute_score(i, others) for i in tqdm(df.text)]
	print('finish one')
	df["price_score"]=[compute_score(i, price) for i in tqdm(df.text)]
	df["mean_price_score"]=[mean_compute_score(i, price) for i in tqdm(df.text)]
	print('finish one')
	df["food_score"]=[compute_score(i, food) for i in tqdm(df.text)]
	df["mean_food_score"]=[mean_compute_score(i, food) for i in tqdm(df.text)]
	print('finish one')
	df["service_score"]=[compute_score(i, service) for i in tqdm(df.text)]
	df["mean_service_score"]=[mean_compute_score(i, service) for i in tqdm(df.text)]

food=['food','chicken','sauce','pizza','meal','lunch','salad','dinner','burger','rice','meat','cheese','sushi','flavor',
	 'beef','taste','pork','soup','breakfast','sandwich','beer','dish','drink','bread','steak','spicy','coffee','shrimp',
	 'selection','dessert','cream','tea','tacos','water','bacon','egg','bbq','ice','fish','wine','course','brunch',
	 'pasta','appetizer','crab','chocalate','salmon','seafood','potato','curry','bite','taco','butter','mac','lobster',
	 'burrito','crispy','cake','salsa','corn','dining','crust','onion','sausage','bland','toast','buffet','pho']
service=['service','time','staff','experience','server','chef','bartender','tender','care','counter','waitress','kind',
		'waiter']
price=['price','cheap','money','deal','tip','size','bill','amount']
environments=['area','place','table','night','location','quality','spot','atmosphere','room','bowl','party','decor','style',
			 'music','patio','street','front','ambiance','seating','parking','glass','evening','space','plate','kitchen']
##time, overalll judgement
others=['restaurant','menu','hour','fan','fun','choice','list','option','delivery']



if __name__ == '__main__':
	path=sys.argv
	df=pd.read_pickle(path[1])
	construct_review_aspect_score(df)
	df.to_pickle(path[2])
	df.to_csv('data/data_experiments.csv')

