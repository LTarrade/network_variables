from multiprocessing import Pool,cpu_count
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import dateutil
import argparse
import logging
import ujson
import glob
import sys
import os

# log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./log/logReg_dataPreparation.log")
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

# args
parser = argparse.ArgumentParser()
parser.add_argument("--path_out", type=str, help="Path to the directory containing the output")
parser.add_argument("--path_users", type=str, help="path to the dataframe containing the values of the network variables for each user")
parser.add_argument("--path_df", type=str, help="path to the dataframe containing informations about the buzzes and changes")
parser.add_argument("--path_idByForm", type=str, help="path to the json file containing the identifiers of words")
parser.add_argument("--path_idByUser", type=str, help="path to the json file containing the identifiers of users")
parser.add_argument("--path_usersByMonthByForm", type=str, help='path to the json file containing the number of total users by month by form')
parser.add_argument("--path_randomWords", type=str, help='path to the directory containing the samples of random words (dataframes)')
parser.add_argument("--path_randomWordsUsers", type=str, help='path to the json file containing the users of random words by month')

args = parser.parse_args()

path_users = args.path_users
path_df = args.path_df
path_idByForm = args.path_idByForm
path_idByUser = args.path_idByUser
path_out = args.path_out
path_usersByMonthByForm = args.path_usersByMonthByForm
path_randomWords = args.path_randomWords
path_randomWordsUsers = args.path_randomWordsUsers


logger.info("Retrieving median values of user network variables for each form (buzz, change, random).")


logger.info("path_users : "+path_users+" ; path_df : "+path_df+" ; path_idByForm : "+path_idByForm+" ; path_idByUser : "+path_idByUser+" ; path_out : "+path_out+" ; path_usersByMonthByForm : "+path_usersByMonthByForm+" ; path_randomWords : "+path_randomWords+" ; path_randomWordsUsers : "+path_randomWordsUsers)


df_users = pd.read_csv(path_users, index_col=0)
df_users.index = df_users.index.astype(str)
df_forms = pd.read_csv(path_df, index_col=0)

idByUser = ujson.load(open(path_idByUser))
idByUser_inv = {str(v):k for k,v in idByUser.items()}
idByForm = ujson.load(open(path_idByForm))
idByForm_inv = {str(v):k for k,v in idByForm.items()}

users_byMonth_byForm = ujson.load(open(path_usersByMonthByForm))
users_byMonth_byForm_id = {}
for form in users_byMonth_byForm : 
	users_byMonth_byForm_id[form] = {}
	for month in users_byMonth_byForm[form] : 
		users_byMonth_byForm_id[form][month] = []
		for user in users_byMonth_byForm[form][month] : 
			users_byMonth_byForm_id[form][month].append(str(idByUser_inv[user]))  

columns = [col for col in df_users.columns if not col.endswith("snap") and not col.startswith("Louvain") and not col.endswith("approx") and not col.endswith("betweennessInCommunity")]
logger.info("variables taken into account : "+str(columns))

# on récupère pour chaque forme la valeur médiane de chacune des variables de réseau observées des utilisateurs qui ont employé cette forme, ainsi que son type (buzz, change, random)
def recup_median(i) : 

	logger.info("Retrieving median values of user network variables for each form (buzz, change, random) - sample "+str(i)+".")

	dic = {}

	# pour les buzz et changements
	for form in df_forms.index : 

		idForm = str(idByForm[form])
		
		users_period = {"innov":set(), "prop":set(), "fix":set()}

		dic[form] = {}
		users = set()

		first_month = df_forms.loc[form,"period"].split(" - ")[0]
		period = int(df_forms.loc[form,"propagation_start"])
		toCheck_innov = [first_month]+sorted([(datetime.strptime(first_month, "%Y-%m")+dateutil.relativedelta.relativedelta(months=j+1)).strftime(r"%Y-%m") for j in range(period-1)])

		period = int(df_forms.loc[form,"propagation_end_ex"])-int(df_forms.loc[form,"propagation_start"])
		toCheck_prop = sorted([(datetime.strptime(toCheck_innov[-1], "%Y-%m")+dateutil.relativedelta.relativedelta(months=j+1)).strftime(r"%Y-%m") for j in range(period)])

		period = 60-int(df_forms.loc[form,"propagation_end_ex"])
		toCheck_fix = sorted([(datetime.strptime(toCheck_prop[-1], "%Y-%m")+dateutil.relativedelta.relativedelta(months=j+1)).strftime(r"%Y-%m") for j in range(period)])
		
		for month in sorted(users_byMonth_byForm_id[idForm]) : 
		
			# global
			for user in users_byMonth_byForm_id[idForm][month] : 
				users.add(user)
			
			# by period
			if month in toCheck_innov :
				for user in users_byMonth_byForm_id[idForm][month] : 
					users_period["innov"].add(user)
				
			if month in toCheck_prop :
				for user in users_byMonth_byForm_id[idForm][month] : 
					# on ne garde que les utilisateurs qui utilisent pour la première fois cette forme pendant cette phase
					if user not in users_period["innov"] :
						users_period["prop"].add(user)
					
			if month in toCheck_fix :
				for user in users_byMonth_byForm_id[idForm][month] : 
					if user not in users_period["innov"] and user not in users_period["prop"]:
						users_period["fix"].add(user)
		
		# global
		sous_df = df_users[df_users.index.isin(users)]
		nb_users = len(users)
		
		for var in columns : 
			# pour le sexe, on récupère plutôt le taux de femmes
			if var == "female" : 
				med = sous_df[var].sum()/sous_df[var].count()
			# pour les autres on récupère la médiane
			else : 
				med = sous_df[var].median()
			dic[form][var] = med	
		dic[form]["type"] = df_forms.loc[form,"type"]
		dic[form]["nb_users"] = nb_users

		# by period
		for p in users_period : 
			sous_df = df_users[df_users.index.isin(users_period[p])]
			nb_users = len(users_period[p])
	
			for var in columns : 
				# pour le sexe, on récupère plutôt le taux de femmes
				if var == "female" : 
					med = sous_df[var].sum()/sous_df[var].count()
				# pour les autres on récupère la médiane
				else : 
					med = sous_df[var].median()
				dic[form][var+"_"+p] = med
			dic[form]["nb_users_"+p] = nb_users

	df_randomWords = pd.read_csv(path_randomWords+"df_randomWords_"+str(i)+".csv", index_col=0)
	users_byMonth_byRandomWord = ujson.load(open(path_randomWordsUsers))

	# puis pour les mots random
	for form in df_randomWords.index : 

		idForm = str(idByForm[form])

		dic[form] = {}
		users = set()

		for month in users_byMonth_byRandomWord[idForm] : 
			for user in users_byMonth_byRandomWord[idForm][month]["users"] : 
				users.add(str(user))

		sous_df = df_users[df_users.index.isin(users)]
		nb_users = len(users)

		for var in columns : 
			# pour le sexe, on récupère plutôt le taux de femmes
			if var == "female" : 
				med = sous_df[var].sum()/sous_df[var].count()
			# pour les autres on récupère la médiane
			else : 
				med = sous_df[var].median()
			dic[form][var] = med	
		dic[form]["type"] = "random"
		dic[form]["nb_users"] = nb_users

	df_varNetwork_medByForm = pd.DataFrame.from_dict(dic,orient="index")
	df_varNetwork_medByForm.to_csv(path_out+"df_varNetwork_medByForm_"+str(i)+".csv")

	logger.info("Retrieving median values of user network variables for each form (buzz, change, random) - sample "+str(i)+" - ended")

try :
	pool = Pool(processes=cpu_count()-2)
	result = pool.map(recup_median, [i for i in range(1,101)])
finally:
	pool.close()
	pool.join()

logger.info("Retrieving median values of user network variables for each form (buzz, change, random) - ended.")

