from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from scipy.stats import fisher_exact
from matplotlib.lines import Line2D
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn import metrics
import seaborn as sn
import pandas as pd
import numpy as np
import argparse
import logging
import ujson

# log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./log/distribAndStats_and_prediction.log")
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument("--path_out", type=str, help="Path to the directory containing the output")
parser.add_argument("--path_df", type=str, help="path to the dataframe containing median values by word and by period")
parser.add_argument("--path_pred", type=str, help="path to the dictionary that contains the prediction results if it already exists", required=False)

args = parser.parse_args()

makePrediction = True

path_df = args.path_df
path_out = args.path_out
if args.path_pred : 
	path_pred = args.path_pred
	makePrediction = False

logger.info("path_df : "+path_df+" ; path_out : "+path_out)

# function that calculates for each period the Kruskal-Wallis statistic for all three distributions and then the Mann-Whitney statistic for each pair of distributions and returns a dictionary containing the results.
def stats_distrib(df_varNetwork_medByForm, period) : 

	columns = ['clusteringCoefficient_networkit', 'PageRank_networkit', 'betweennessInComm_toScale', 'lengthMean_rw']
	columns_period = [col+"_"+period for col in columns]

	# We assign to the control words their global value in period p
	temp1 = df_varNetwork_medByForm[~(df_varNetwork_medByForm.type=="random")]
	temp2 = df_varNetwork_medByForm[df_varNetwork_medByForm.type=="random"]
	for col in columns_period : 
		temp2[col]=temp2[col[:-len("_"+period)]]
	df_varNetwork_medByForm = pd.concat([temp1,temp2])

	sous_df = df_varNetwork_medByForm[columns_period]
	sous_df["type"] = df_varNetwork_medByForm["type"]

	results_stats = {}

	for col in columns_period : 

		results_stats[col] = {}

		results_stats[col]["kruskal"] = (kruskal(sous_df[sous_df.type=="random"][col],
			sous_df[sous_df.type=="buzz"][col],
			sous_df[sous_df.type=="change"][col]))

		results_stats[col]["mann-whitney"] = {"random-change":{}, "random-buzz":{}, "change-buzz":{}}

		for pair in results_stats[col]["mann-whitney"] : 

			first_type, second_type = pair.split("-")[0], pair.split("-")[1]
			results_stats[col]["mann-whitney"][pair] = {}

			for alternative in ["two-sided", "greater", "less"] : 
				
				results_stats[col]["mann-whitney"][pair][alternative]= mannwhitneyu(sous_df[sous_df.type==first_type][col],
					sous_df[sous_df.type==second_type][col],alternative=alternative)


	return results_stats


df = pd.read_csv(path_df, index_col=0)
period = ["innov", "prop", "fix"]

logger.info('Computes statistics.')
dic = {}

for p in period : 

	dic[p]=stats_distrib(df, p)

ujson.dump(dic, open(path_out+"12_results_stats.json", "w"))

logger.info('Computes statistics ended. Results saved in '+path_out+" 12_results_stats.json")


logger.info('Visualisation of distributions.')


# from the dataframe of median values per word, we create a dataframe with each entry representing a word+period (mot1_innov, mot2_innov, .... mot1_prop, mot2_prop, etc.)
columns = ['clusteringCoefficient_networkit', 'PageRank_networkit', 'betweennessInComm_toScale', 'lengthMean_rw']
allDfToConcat = []

for col in columns : 
	
	dfToConcat_col = []
	
	for p in period : 

		temp_innov = df[~(df.type=="random")][col+"_"+p].to_frame()
		temp_random = df[df.type=="random"][col].to_frame()
		temp_innov = temp_innov.rename(columns={col+"_"+p:col})

		df_temp = pd.concat([temp_innov,temp_random], axis=0)
		df_temp["type"] = df_temp.index.map(df.type)
		df_temp.index = df_temp.index+"_"+p
		
		dfToConcat_col.append(df_temp)
		
	df_col = pd.concat(dfToConcat_col)
	typeDf = df_col.type.to_frame()
	
	allDfToConcat.append(df_col[col].to_frame())
	
new_df = pd.concat(allDfToConcat, axis=1)
new_df["type"] = typeDf.index.map(typeDf.type)
new_df["period"] = [name.split("_")[1] for name in new_df.index]

fig, ax = plt.subplots(3, 4, figsize=[24,24])
sn.set_theme()
nb = 0 

# to adjust the scales on the graph
dic_lim = {'clusteringCoefficient_networkit': (0, 0.22), 'PageRank_networkit': (0.1e-07, 4.5e-07), 'betweennessInComm_toScale': (-0.4, 2), 'lengthMean_rw': (2, 12)}

legend = {"innov":"Innovation", "prop":"Propagation", "fix":"Fixation", "clusteringCoefficient_networkit" : "Coefficient de clustering", "PageRank_networkit":"Score de PageRank", "betweennessInComm_toScale":"Centralit?? au sein de la communaut??", "lengthMean_rw":"Nombre de pas moyen\npour sortir de la communaut??"}
colors = {"change":"#4282B3", "buzz":"#819E57", "random":"#F5A614"}

for i,p in enumerate(period) : 
		
	for j,col in enumerate(columns) : 
		
		sn.boxplot(data=new_df[new_df.period==p], y=col, x="type", color='#eaeaf2', width=0.6, showfliers=False, notch=True, ax=ax[i][j])
		sn.stripplot(data=new_df[new_df.period==p], y=col, x="type", palette=colors, size=4, alpha=0.8, ax=ax[i][j])
		
		ax[i][j].set_ylim(dic_lim[col])
		
		if i==0 :
			ax[i][j].set_title(legend[col]+"\n", fontsize=18)
		if j==0 :
			ax[i][j].set_ylabel(legend[p]+"\n", fontsize=18)
		else : 
			ax[i][j].set_ylabel(legend[p], visible=False)
	
		ax[i][j].set_xticks([])
		ax[i][j].set_xlabel("", visible=False)
		
plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(hspace=0.04)

legend_elements = [Line2D([0], [0], marker='o', color="#eaeaf2", label='Changements',
						  markerfacecolor=colors["change"], markersize=8),
				   Line2D([0], [0], marker='o', color='#eaeaf2', label='Buzz',
						  markerfacecolor=colors["buzz"], markersize=8),
				   Line2D([0], [0], marker='o', color='#eaeaf2', label='Mots t??moins',
						  markerfacecolor=colors["random"], markersize=8)]
ax[i][j].legend(handles=legend_elements, loc='lower right', fontsize=16)

plt.savefig(path_out+"12_distrib.png", format="png")

logger.info("Visualisation of distributions ended. Image saved in "+path_out+"distrib.png.")


if makePrediction :

	logger.info("Prediction on lexical innovations")

	dic = {"AUC":{}, "PREC":{}, "ROC":{}, "CONF":{}, "ODDS":{}}

	for p in period[:2] : 

		columns_period = [col+"_"+p for col in columns] 

		prec = []
		conf = []
		odds = []
		auc = []
		y_tests = []
		y_preds = []

		for i in range(10000) : 
			
			df_innovLex = df[df.type.isin(["change","buzz"])]

			# the number of buzzes is adjusted in relation to the changes
			df_eq = df_innovLex[df_innovLex.type=="change"]
			nb_toKeep = len(df_innovLex[df_innovLex.type=="change"])
			temp1 = df_innovLex[df_innovLex.type=="buzz"].sample(n=nb_toKeep, replace=False)
			df_eq = pd.concat([df_eq,temp1])
			df_innovLex = df_eq

			# we select the data of observed period and the data are standardised
			sous_df = df_innovLex[columns_period]
			df_CR = pd.DataFrame(RobustScaler().fit_transform(sous_df), index=sous_df.index, columns=sous_df.columns)

			# change=0, buzz=1
			df_CR["type"] = df_innovLex["type"]
			df_CR["type"]=df_CR["type"].replace(["change"],0)
			df_CR["type"]=df_CR["type"].replace(["buzz"],1)

			y = df_CR.type
			x = df_CR.drop(["type"], axis=1)

			# we divide into training and test data (75-25)
			x_train, x_test, y_train, y_test = train_test_split(x, y)
			len(x_train), len(x_test), len(y_test)

			# we train the model
			modele_regLog = linear_model.LogisticRegression(solver='liblinear', multi_class='auto')
			modele_regLog.fit(x_train,y_train)

			# the accuracy is recovered
			precision = modele_regLog.score(x_test,y_test)
			prec.append(precision)

			# we get the odd coef of the different values (order: x.columns)
			odd = np.exp(modele_regLog.coef_[0])
			odds.append(odd)

			# the confusion matrix is recovered
			y_pred = modele_regLog.predict(x_test)
			y_tests.append(y_test)
			y_preds.append(y_pred)
			cm = metrics.confusion_matrix(y_test, y_pred)
			conf.append(cm)
			
			# the area under the ROC curve is recovered
			y_pred_proba = modele_regLog.predict_proba(x_test)[::,1]
			roc = metrics.roc_auc_score(y_test, y_pred_proba)
			auc.append(roc)

			conf_list = []
			for c in conf : 
			    conf_list.append([[int(nb) for nb in e] for e in c])

			odds_list = [list(o) for o in odds]

			dic["AUC"][p]=auc
			dic["PREC"][p]=prec
			dic["ROC"][p]=roc
			dic["CONF"][p]=conf_list
			dic["ODDS"][p]=odds_list

	ujson.dump(dic, open(path_out+"12_results_prediction.json", "w"))

	logger.info("Prediction on lexical innovations ended. Results saves in "+path_out+"12_results_prediction.json")

else : 

	dic = ujson.load(open(path_pred))

logger.info("plot AUC and precision distributions")
# plot precision and AUC distributions
def plot_eval(e) : 

	colors = ["#5B996C", "#3787A0"]
	plt.figure(figsize=[12,4])
	
	for i,p in enumerate(period[:2]) : 
		data = dic[e][p]

		sn.histplot(data, stat="percent",element="bars", color=colors[i], zorder=1, alpha=0.6)
		plt.xlim([0,1])
		plt.xlabel(e, fontsize=14)
		plt.ylabel("Pourcentage", fontsize=14)
		plt.axvline(x=np.mean(data), linewidth=3, color=colors[i], linestyle="--", zorder=1, label=e+" mean - "+legend[p]+" : "+str(round(np.mean(data),2)))
		plt.legend(fontsize=13)


	plt.savefig(path_out+"12_"+e+".png", format='png')

# to compute Fisher exact test on confusion matrix
pval_fisher = {}
for p in period[:2] : 
	pval_fisher[p] = []
	for i,m in enumerate(dic["CONF"][p]) :
	    oddsratio, pvalue = fisher_exact(m, alternative='two-sided')
	    pval_fisher[p].append(pvalue)

ujson.dump(pval_fisher, open(path_out+"12_pval_fisher.json", "w"))

plot_eval("AUC")
plot_eval("PREC")

logger.info("ended.")
