## the reviewer features include: 
##'user_average_stars', 'user_fans',
## 'user_friends', 'user_received_funny', 'user_received_cool',
##     'user_received_useful', 'user_days_since','star_deviation'
import pandas as pd 
import sys
from datetime import datetime


def construc_reviewers_feature(df):
    df["user_average_stars"]=df.user_id.map(df_ur.set_index('user_id').average_stars)
    df["user_fans"]=df.user_id.map(df_ur.set_index('user_id').fans)
    df['user_friends']=df.user_id.map(df_ur.set_index('user_id').friends)
    df['user_received_funny']=df.user_id.map(df_ur.set_index('user_id').funny)
    df['user_received_cool']=df.user_id.map(df_ur.set_index('user_id').cool)
    df['user_received_useful']=df.user_id.map(df_ur.set_index('user_id').useful)
    


if __name__ == '__main__':
	print('start')
	df_ur=pd.read_csv('data/user.csv')
	path=sys.argv
	df=pd.read_pickle(path[1])
	print("finish loading data")
	print('start to generate features')
	construc_reviewers_feature(df)
	date_format = "%Y-%m-%d"
	b = datetime.strptime('2017-7-26', date_format)
	df_ur['days_since']= [b-datetime.strptime(i, date_format) for i in df_ur.yelping_since]
	df['user_days_since']=df.user_id.map(df_ur.set_index('user_id').days_since)
	df['star_deviation']=df.stars-df['business_avg_star']
	print('finish')
	df.to_pickle(path[2])
