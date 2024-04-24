# coding:utf-8
from sklearn.preprocessing import RobustScaler
from multiprocessing import Pool,cpu_count
import networkit as nk
import pandas as pd
import argparse
import logging
import ujson
import ast
import sys
import os

#log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./log/network_var_networkit.log")
handler.setFormatter(logging.Formatter("%(asctime)s; %(levelname)s; %(message)s"))
logger.addHandler(handler)

# args
parser = argparse.ArgumentParser()
parser.add_argument("--path_edges", type=str, help="Path to the file contains all the links between the users of the corpus.")
parser.add_argument("--path_idUsers", type=str, help="Path to json files containing the id and corresponding users.")
parser.add_argument("--path_out", type=str, help="Path to the directory containing the output.")
parser.add_argument("--path_users", type=str, help="Path to the directory containing the dataframe of users.")

args = parser.parse_args()

path_edges = args.path_edges
path_idUsers = args.path_idUsers
path_out = args.path_out
path_users = args.path_users

logger.info("path_edges : "+path_edges+" ; path_idUsers : "+path_idUsers+" ; path_out : "+path_out+" ; paht_users : "+path_users)

# Function that calculates the centrality within the community c
def recupVarInComm(c) : 

	logger.info("Computing betweenness in community for users of community n°"+str(c))

	sous_df = df_users[df_users.LouvainCommunity_networkit==c]
	usersInComm = sous_df.index.tolist()
	usersInComm = [int(u) for u in usersInComm]
	subGraph = nk.graphtools.subgraphFromNodes(G, usersInComm)
	
	
	#transformation to undirected graph
	G_undirected = nk.graphtools.toUndirected(subGraph)
	G_undirected.removeMultiEdges()

	if len(sous_df) < 10000 :
		
		btw = nk.centrality.Betweenness(subGraph,normalized=True)
		btw.run()

		out = open(path_out+"comm_"+str(c)+"_btw_ranking.txt", "w")
		for e in btw.ranking() : 
				out.write(str(e)+"\n")
		out.close()

	else :

		kadabra = nk.centrality.KadabraBetweenness(G_undirected, err=0.0001, delta=0.1)
		kadabra.run()

		out = open(path_out+"comm_"+str(c)+"_KadabraBetweenness_err0001_delta1_ranking.txt", "w")
		for e in kadabra.ranking() : 
				out.write(str(e)+"\n")
		out.close()

	logger.info("Computing betweenness in community for users of community n°"+str(c)+" - ended.")


# loading the directed graph of the whole corpus
G = nk.graphio.SNAPGraphReader(directed=True, remapNodes=False, nodeCount=2585663).read(path_edges)
logger.info("Graph loading complete, %s nodes, %s ties."%(str(G.numberOfNodes()), str(G.numberOfEdges())))


logger.info("KadabraBetweenness centrality in communities calculation")

df_users = pd.read_csv(path_users,index_col=0)

comms = df_users.LouvainCommunity_networkit.dropna().tolist()
comms = list(set(comms))

btw_folder = path_out+"09_btwByComm/"
if not os.path.exists(btw_folder):
	os.makedirs(btw_folder)

try :
	pool = Pool(processes=cpu_count()-2)
	results = pool.map(recupVarInComm, comms)
finally:
	pool.close()
	pool.join()

logger.info("KadabraBetweenness centrality in communities calculation completed.")


logger.info("adding results to the dataframe.")

dic_betweennessInCommunity = {}
dic_betweennessInCommunity_approx = {}

files = glob.glob(btw_folder+"*Kadabra*")

for file in files :
        
    file = open(file)
    
    for line in file :
        l = ast.literal_eval(line)
        user = str(l[0])
        score = l[1]

        dic_betweennessInCommunity[user]=score
        dic_betweennessInCommunity_approx[user]=True
        

files = glob.glob(btw_folder+"*btw*")

for file in files : 
        
    file = open(file)
    
    for line in file :
        try : 
            l = ast.literal_eval(line)
        except : 
            l = line.rstrip()[1:-1].split(", ")
            
        user = str(l[0])
        score = l[1]
        if score=="nan" : 
            score = None
        
        dic_betweennessInCommunity[user]=score
        dic_betweennessInCommunity_approx[user]=False
        
df_users["betweennessInCommunity"] = df_users.index.map(dic_betweennessInCommunity)
df_users["betweennessInCommunity_approx"] = df_users.index.map(dic_betweennessInCommunity_approx)

logger.info("adding results to the dataframe completed.")


logger.info("scaling of centrality measures.")

dic = {}
for c in comms :
    try : 
        sous_df = df_users[df_users.LouvainCommunity_networkit==c][["betweennessInCommunity"]]
        sous_df_CR = pd.DataFrame(RobustScaler().fit_transform(sous_df), index=sous_df.index, columns=sous_df.columns)
        for user in sous_df_CR.index : 
            dic[user]=sous_df_CR.loc[user,"betweennessInCommunity"]
    except : 
        print("\n"+str(c)+"\n")

df_users["betweennessInComm_toScale"] = df_users.index.map(dic)

logger.info("scaling of centrality measures completed.")

# adapt in function
nameDF = "nameDF"
df_users.to_csv(path_out+nameDF+".csv")

logger.info("Ended.")

