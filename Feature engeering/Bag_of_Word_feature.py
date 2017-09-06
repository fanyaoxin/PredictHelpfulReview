import pandas 


from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy
from sklearn.datasets import load_digits
from scipy.stats import randint as sp_randint
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sys
import pandas as pd




if __name__ == '__main__':
	path=sys.argv
	df_use=pd.read_pickle(path[2])
	df=pd.read_pickle(path[1])
	print('start training')
	text_clf_MNB = Pipeline([('vect', text.CountVectorizer()),  
					('tfidf',text.TfidfTransformer()),
					('clf',MultinomialNB()),])
	X_train, X_test, y_train, y_test = train_test_split(
    	df_use.text, df_use.label, test_size=0.33, random_state=42)
	text_clf_MNB.fit(X_train,y_train)
	print('predict BOW probability')
	pred=text_clf_MNB.predict_proba(df.text)
	df['BOW']=0
	df['BOW']=pred
	df.to_pickle(path[3])

