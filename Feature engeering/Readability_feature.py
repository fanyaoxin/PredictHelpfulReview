

from textstat.textstat import textstat
import re
import sys
import language_check
tool = language_check.LanguageTool('en-US')
import math
from tqdm import tqdm
import pandas as pd
def compute_syllables(text):
	num_sentence=textstat.sentence_count(text)
	text=re.sub('[^A-Za-z0-9]+', ' ', text)
	word_list=text.split()
	num_simple=0
	num_complex=0
	num_syllables=0
	for i in word_list:
		try:
			syllables=nsyl(i)
			if syllables>=3:
				num_complex+=1
				num_syllables=num_syllables+syllables
			else:
				num_simple+=1
				num_syllables=num_syllables+syllables
		except:
			continue
	return [num_simple,num_complex,num_syllables,num_sentence]

def construct_readability_feature(df):
	df['simple_words']=[compute_syllables(i)[0] for i in tqdm(df.text)]
	df['complex_words']=[compute_syllables(i)[1] for i in tqdm(df.text)]
	df['syllables']=[compute_syllables(i)[2] for i in tqdm(df.text)]
	df['num_sentence']=[compute_syllables(i)[3] for i in tqdm(df.text)]
	df['error_num']=[len(tool.check(i)) for i in tqdm(df.text)]
	df['error_num']=df['error_num']-df['num_sentence']
	df['fog_index']=0.4*((df.simple_words+df.complex_words)/df.num_sentence+100*(df.complex_words/df['review_all_length']))
	df['Flesch_reading_ease_score']=206.835-1.015*((df.simple_words+df.complex_words)/df.num_sentence)-84.6*(df.syllables/(df.simple_words+df.complex_words))
	df['SMOG_index']=[1.043*math.sqrt(float(30*((df.simple_words[i]+df.complex_words[i])/df.num_sentence[i])))+3.1219 for i in df.index]
	df['fog_index_simple']=0.4*((df.simple_words+df.complex_words)/df.num_sentence+100*(df.simple_words/df['review_all_length']))
	df['word_length']=df['syllables']/df['review_all_length']

if __name__ == '__main__':
	path=sys.argv
	df=pd.read_pickle(path[1])
	construct_readability_feature(df)
	df.to_pickle(path[2])



