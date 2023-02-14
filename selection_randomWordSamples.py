# coding: utf-8
from multiprocessing import Pool, cpu_count
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
import re

# log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./log/selection_randomWordSamples.log")
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

# args
parser = argparse.ArgumentParser()
parser.add_argument("--path_tokenizedTweets", type=str, help="Path to the directory of tokenized tweets")
parser.add_argument("--path_out", type=str, help="Path to the directory containing the output")
parser.add_argument("--path_users", type=str, help="path to the dataframe containing the values of the network variables for each user")
parser.add_argument("--path_df", type=str, help="path to the dataframe containing informations about the buzzes and changes")
parser.add_argument("--path_idByForm", type=str, help="path to the json file containing the identifiers of words")
parser.add_argument("--path_idByUser", type=str, help="path to the json file containing the identifiers of users")
parser.add_argument("--path_usersByMonth", type=str, help='path to the json file containing the number of total users by month')
parser.add_argument("--path_occForms", type=str, help='path to the json file containing the total number of occurrences of all forms by month')


args = parser.parse_args()

path_tokenizedTweets = args.path_tokenizedTweets
path_users = args.path_users
path_df = args.path_df
path_idByForm = args.path_idByForm
path_idByUser = args.path_idByUser
path_out = args.path_out
path_usersByMonth = args.path_usersByMonth
path_occForms = args.path_occForms

first_month = "2013-02"
period = 60
nbWords = 200

logger.info("path_tokenizedTweets : "+path_tokenizedTweets+" ; path_users : "+path_users+" ; path_df : "+path_df+" ; path_idByForm : "+path_idByForm+" ; path_idByUser : "+path_idByUser+" ; path_out : "+path_out+" ; path_usersByMonth : "+path_usersByMonth+" ; path_occForms : "+path_occForms+" ; first_month : "+first_month+" ; period : "+str(period)+" ; nbWords : "+str(nbWords))

df_users = pd.read_csv(path_users, index_col=0)
df_users.index = df_users.index.astype(str)
df_forms = pd.read_csv(path_df, index_col=0)

idByUser = ujson.load(open(path_idByUser))
idByUser_inv = {str(v):k for k,v in idByUser.items()}
idByForm = ujson.load(open(path_idByForm))
idByForm_inv = {str(v):k for k,v in idByForm.items()}

months = [first_month]+sorted([(datetime.strptime(first_month, "%Y-%m")+dateutil.relativedelta.relativedelta(months=i+1)).strftime(r"%Y-%m") for i in range(period-1)])
filesToTreat = []
for m in months : 
    filesToTreat+=glob.glob(path_tokenizedTweets+m+"*")
filesToTreat = sorted(filesToTreat)

# we selected only the forms with at less 100 occurrences
occForms = ujson.load(open(path_occForms))
more100 = {f for f in occForms if occForms[f]>100}

print("Retrieving the number of occurrences and users of all words.\n")

def nbOccAndUsers_retrieval(f, filterList) :

    nbOccAndUsers = {}

    fileName = os.path.basename(f).split("_tokenized")[0]

    logger.info("Retrieval of the number of occurrences and users of words for the tweets contained in the file %s."%fileName)

    file = open(f)

    for line in file :

        tweet = ujson.loads(line.rstrip())
        user = tweet["user_id"]
        user = int(idByUser_inv[user])
        
        nb_sent = tweet["nb_sent"]

        for sent in range(1,nb_sent+1) : 
            for token in tweet["tokenization"]["sentence_"+str(sent)]["tokens"] : 
                form = token.lower()
                if not form.startswith("@") and not form.startswith("http") and not form.startswith("www.") and re.match(r"(\w|-|')+", form):
                    id_form = str(idByForm[form])
                    
                    if id_form in filterList :
                        
                        if id_form not in nbOccAndUsers : 
                            nbOccAndUsers[id_form]={"nbOcc":0, "users":set()}
                            
                        nbOccAndUsers[id_form]["nbOcc"]+=1
                        nbOccAndUsers[id_form]["users"].add(user)

    logger.info("Retrieval of the number of occurrences and users of words for the tweets contained in the file %s - ended."%fileName)

    return nbOccAndUsers


logger.info("Retrieving the number of occurrences and users of all words for the period "+months[0]+" - "+months[-1]+".")

nbOccAndUsers_byMonth = {}

for m in sorted(months) :
    
    nbOccAndUsers_byMonth[m]={}
    
    logger.info("Process of month %s"%m)

    try :
        pool = Pool(processes=cpu_count()-2)
        result = pool.starmap(nbOccAndUsers_retrieval, [(f,more100) for f in filesToTreat if os.path.basename(f).startswith(m)])
    finally:
        pool.close()
        pool.join()
        
    for i,r in enumerate(result) :

        nbOccAndUsers = r

        for form in nbOccAndUsers : 

            if form not in nbOccAndUsers_byMonth[m] : 
                nbOccAndUsers_byMonth[m][form] = {"nbOcc":0, "users":set()}

            nbOccAndUsers_byMonth[m][form]["nbOcc"]+=nbOccAndUsers[form]["nbOcc"]
            for u in nbOccAndUsers[form]["users"] : 
                nbOccAndUsers_byMonth[m][form]["users"].add(u)    
                
    for form in nbOccAndUsers_byMonth[m] :
        nbOccAndUsers_byMonth[m][form]["users"]=len(nbOccAndUsers_byMonth[m][form]["users"])
        
    logger.info("Process of month %s ended"%m)
    
logger.info("Retrieving the number of occurrences and users of all words for the period "+months[0]+" - "+months[-1]+" ended.")

logger.info("Saving results in "+path_out)
ujson.dump(nbOccAndUsers_byMonth, open(path_out+"10_nbOccAndUsers_more100_byMonth_201302_201801.json", "w"))

logger.info("Ended.")


#######################################################################################################################################
print("Retrieving the number of occurrences and users of all words - ended.\n")
print("Selection of forms with a low variance in usage rates.\n")
#######################################################################################################################################


logger.info("Selection of forms with a low variance in usage rates.")

# on récupère dans des dataframes distincts l'ensemble du nombre d'occurrences et d'utilisateurs par mois 
nbOccAndUsers_byMonth = ujson.load(open(path_out+"10_nbOccAndUsers_more100_byMonth_201302_201801.json"))

nbUserByMonthByForm = {}
nbOccByMonthByForm = {}
for i,month in enumerate(nbOccAndUsers_byMonth) :
    sys.stdout.write("\r"+month)
    for form in nbOccAndUsers_byMonth[month] :
        if form not in nbUserByMonthByForm :
            nbUserByMonthByForm[form]={m:None for m in sorted(months)}
            nbOccByMonthByForm[form]={m:None for m in sorted(months)}
        nbUserByMonthByForm[form][month]=nbOccAndUsers_byMonth[month][form]["users"]
        nbOccByMonthByForm[form][month]=nbOccAndUsers_byMonth[month][form]["nbOcc"]

df_occByFormByMonth = pd.DataFrame.from_dict(nbOccByMonthByForm, orient="index")
df_nbUsersByFormByMonth = pd.DataFrame.from_dict(nbUserByMonthByForm, orient="index")
df_occByFormByMonth = df_occByFormByMonth.rename(index={ind:idByForm_inv[ind] for ind in df_occByFormByMonth.index})
df_nbUsersByFormByMonth = df_nbUsersByFormByMonth.rename(index={ind:idByForm_inv[ind] for ind in df_nbUsersByFormByMonth.index})

usersByMonth = ujson.load(open(path_usersByMonth))
nbUsersByMonth = {}
for m in usersByMonth : 
    if m in months : 
        nbUsersByMonth[m] = len(usersByMonth[m])
df_months = pd.DataFrame.from_dict(nbUsersByMonth,orient="index").transpose()

# on calcule le taux d'utilisation relatif
df_nbUsersByFormByMonth_rate = pd.DataFrame(columns=df_nbUsersByFormByMonth.columns)
df_nbUsersByFormByMonth_rate = df_nbUsersByFormByMonth/df_months.loc[0]

df_nbUsersByFormByMonth_rate_rel = df_nbUsersByFormByMonth_rate.div(df_nbUsersByFormByMonth_rate.sum(axis=1), axis=0)

# on récupère pour chaque mot l'écart-type (racine carrée de la variance, + interpretable) de ses taux d'utilisation relatifs pendant les 60 mois, et on ne sélectionne que ceux ayant peu de variance pour avoir des courbes relativement stables (donc pas similaires aux buzzs ou changements)
wordsWithoutVar = []
threshold = 0.007
for i,word in enumerate(df_nbUsersByFormByMonth_rate_rel.index) :
    nbNan = np.sum(df_nbUsersByFormByMonth_rate_rel.loc[word].isna())
    std = df_nbUsersByFormByMonth_rate_rel.loc[word].std()
    # ici on vérifie que la variance ne dépasse pas un certain seuil, qu'il y ait au mois autant de mois avec des valeurs que le nombre de mois minim. dans les buzz/changements, et qu'il ne s'agisse pas d'adresses mail 
    if std < threshold and nbNan<=38 and not re.match(".*@.+\..{2,3}",word): 
        wordsWithoutVar.append(word)

to_save = []
for w in wordsWithoutVar : 
    to_save.append(str(idByForm[w]))

logger.info("Saving results in : "+path_out)
ujson.dump(to_save,open(path_out+'10_words_filtered_forRandomWords.txt', 'w'))

logger.info("Selection of forms with a low variance in usage rates ended.")

logger.info("Ended.")


#######################################################################################################################################
print("Selection of forms with a low variance in usage rates - ended.\n")
print("Retrieving the number of total users of filtered words.\n")
#######################################################################################################################################


words_filtered = ujson.load(open(path_out+'words_filtered_forRandomWords.txt'))
words_filtered = set(words_filtered)

def nbTotalUsers_retrieval(f) :

    nbTotalUsers = {}

    fileName = os.path.basename(f).split("_tokenized")[0]

    logger.info("Retrieval of the users of filtered words for the tweets contained in the file %s."%fileName)

    file = open(f)

    for line in file :

        tweet = ujson.loads(line.rstrip())
        user = tweet["user_id"]
        user = int(idByUser_inv[user])
        
        nb_sent = tweet["nb_sent"]

        for sent in range(1,nb_sent+1) : 
            for token in tweet["tokenization"]["sentence_"+str(sent)]["tokens"] : 
                form = token.lower()
                if not form.startswith("@") and not form.startswith("http") and not form.startswith("www.") and re.match(r"(\w|-|')+", form):
                    id_form = str(idByForm[form])
                    
                    if id_form in words_filtered :
                        
                        if id_form not in nbTotalUsers : 
                            nbTotalUsers[id_form]=set()
                            
                        nbTotalUsers[id_form].add(user)

    logger.info("Retrieval of the users of filtered words for the tweets contained in the file %s - ended."%fileName)

    return nbTotalUsers


logger.info("Retrieving the number of total users of filtered words for the period "+months[0]+" - "+months[-1]+".")

nbTotalUsers_byMonth = {}

for m in sorted(months) :
    
    nbTotalUsers_byMonth[m]={}
    
    logger.info("Process of month %s"%m)

    try :
        pool = Pool(processes=cpu_count()-2)
        result = pool.map(nbTotalUsers_retrieval, [f for f in filesToTreat if os.path.basename(f).startswith(m)])
    finally:
        pool.close()
        pool.join()
        
    for i,r in enumerate(result) :

        nbTotalUsers = r

        for form in nbTotalUsers : 

            if form not in nbTotalUsers_byMonth[m] : 
                nbTotalUsers_byMonth[m][form] = set()

            for u in nbTotalUsers[form] : 
                nbTotalUsers_byMonth[m][form].add(u)    
        
    logger.info("Process of month %s ended"%m)

logger.info("Retrieving the number of total users of filtered words for the period "+months[0]+" - "+months[-1]+" ended.")

logger.info("Saving results in "+path_out)
nbTotalUsers = {} 
for month in nbTotalUsers_byMonth :
    for form in nbTotalUsers_byMonth[month] :
        if form not in nbTotalUsers : 
            nbTotalUsers[form]=set()
        for user in nbTotalUsers_byMonth[month][form] :
            nbTotalUsers[form].add(user)

for form in nbTotalUsers :
    nbTotalUsers[form]=len(nbTotalUsers[form])

ujson.dump(nbTotalUsers, open(path_out+"10_nbTotalUsers_filteredWords_byMonth_201302_201801.json", "w"))

logger.info("Ended.")


#######################################################################################################################################
print("Retrieving the number of total users of filtered words - ended.\n")
print("Selection of 100 samples of 200 random words.\n")
#######################################################################################################################################


logger.info("Selection of 100 samples of 200 random words each, with a distribution similar to that of the linguistic innovations in terms of number of users.")

nbTotalUsers_byFilteredForm = ujson.load(open(path_out+"10_nbTotalUsers_filteredWords_byMonth_201302_201801.json"))

df_nbTotalUsers_byFilteredForm = pd.DataFrame.from_dict(nbTotalUsers_byFilteredForm, orient="index")
df_nbTotalUsers_byFilteredForm.rename(index=idByForm_inv,inplace=True)
df_nbTotalUsers_byFilteredForm.rename(columns={0:"nbTotalUsers"},inplace=True)

df_list = []
for i in range(100) :
    
    randomWords = df_nbTotalUsers_byFilteredForm[~(df_nbTotalUsers_byFilteredForm.index.isin(df_forms.index)) & (df_nbTotalUsers_byFilteredForm.nbTotalUsers>df_forms.nbUsers_period.quantile(0.8)) & (df_nbTotalUsers_byFilteredForm.nbTotalUsers<=df_forms.nbUsers_period.max())].sample(n=round(nbWords/5),  replace=False).index.tolist()
    randomWords += df_nbTotalUsers_byFilteredForm[~(df_nbTotalUsers_byFilteredForm.index.isin(df_forms.index)) & (df_nbTotalUsers_byFilteredForm.nbTotalUsers>df_forms.nbUsers_period.quantile(0.6)) & (df_nbTotalUsers_byFilteredForm.nbTotalUsers<=df_forms.nbUsers_period.quantile(0.8))].sample(n=round(nbWords/5),  replace=False).index.tolist()
    randomWords += df_nbTotalUsers_byFilteredForm[~(df_nbTotalUsers_byFilteredForm.index.isin(df_forms.index)) & (df_nbTotalUsers_byFilteredForm.nbTotalUsers>=df_forms.nbUsers_period.min()) & (df_nbTotalUsers_byFilteredForm.nbTotalUsers<=df_forms.nbUsers_period.quantile(0.6))].sample(n=round((nbWords/5)*3),  replace=False).index.tolist()

    df_randomWords = df_nbTotalUsers_byFilteredForm[df_nbTotalUsers_byFilteredForm.index.isin(randomWords)]
    df_list.append(df_randomWords)

randomWords = set()
for i,df in enumerate(df_list) :
    df.to_csv(path_out+"10_randomWords_samples/df_randomWords_"+str(i+1)+".csv") 
    for form in df.index :
        randomWords.add(form)

logger.info("Selection of 100 samples of 200 random words each, with a distribution similar to that of the linguistic innovations in terms of number of users - ended.")

logger.info("Ended.")


#######################################################################################################################################
print("Selection of 100 samples of 200 random words - ended.\n")
print("Retrieve number of occurrences and users by month for random words.\n")
#######################################################################################################################################

logger.info("Retrieve number of occurrences and users by month for random words.")

nbOccAndUsers_byMonth = {}

for m in sorted(months) :
        
    logger.info("Process of month %s"%m)

    try :
        pool = Pool(processes=cpu_count()-2)
        result = pool.starmap(nbOccAndUsers_retrieval, [(f,randomWords) for f in filesToTreat if os.path.basename(f).startswith(m)])
    finally:
        pool.close()
        pool.join()
        
    for i,r in enumerate(result) :

        nbOccAndUsers = r

        for form in nbOccAndUsers : 

            if form not in nbOccAndUsers_byMonth : 
                nbOccAndUsers_byMonth[form] = {}
            
            if m not in nbOccAndUsers_byMonth[form] : 
                nbOccAndUsers_byMonth[form][m] = {"nbOcc":0, "users":set()}

            nbOccAndUsers_byMonth[form][m]["nbOcc"]+=nbOccAndUsers[form]["nbOcc"]
            for u in nbOccAndUsers[form]["users"] : 
                nbOccAndUsers_byMonth[form][m]["users"].add(u)    
                
    for form in nbOccAndUsers_byMonth :
        if m in nbOccAndUsers_byMonth[form] :
            nbOccAndUsers_byMonth[form][m]["users"]=list(nbOccAndUsers_byMonth[form][m]["users"])
        
    logger.info("Process of month %s ended"%m)

logger.info("Retrieve number of occurrences and users by month for random words - ended.")
    
logger.info("Saving results in "+path_out)
ujson.dump(nbOccAndUsers_byMonth, open(path_out+"10_nbOccAndUsers_randomWords_byMonth_201302_201801.json", "w"))

logger.info("Ended.")

print("Retrieve number of occurrences and users by month for random words - ended.\n")
