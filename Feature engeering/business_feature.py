import sys
import pandas as pd
import time

##the business_features include: 'business_star',
##'business_review_count', 'city', 'business_avg_star'


def construct_business_feature(df):
	df["business_star"]=df.business_id.map(df_bu.set_index('business_id').stars)
	df["business_review_count"]=df.business_id.map(df_bu.set_index('business_id').review_count)
	df['city']=df.business_id.map(df_bu.set_index('business_id').city)

if __name__ == '__main__':
	start=time.clock()
	print('start ')
	df_review_all=pd.read_csv('data/review_all.csv')
	path=sys.argv
	df=pd.read_pickle(path[1])
	print('finish loading reivew_all')
	print('time use:',float(time.clock()-start))
	print('start generating features')
	df['business_avg_star']=df.business_id.map(df_review_all.groupby('business_id').stars.mean())
	del df_review_all
	df_bu=pd.read_csv('data/business.csv')
	construct_business_feature(df)
	df.to_pickle(path[2])
	print('time use:',float(time.clock()-start))