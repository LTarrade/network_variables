# coding:utf-8
import networkit as nk
import argparse
import logging
import ujson
import sys

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

args = parser.parse_args()

path_edges = args.path_edges
path_idUsers = args.path_idUsers
path_out = args.path_out

logger.info("path_edges : "+path_edges+" ; path_idUsers : "+path_idUsers+" ; path_out : "+path_out)

# loading the directed graph of the whole corpus
G = nk.graphio.SNAPGraphReader(directed=True, remapNodes=False, nodeCount=2585663).read(path_edges)
logger.info("Graph loading complete, %s nodes, %s ties."%(str(G.numberOfNodes()), str(G.numberOfEdges())))


logger.info("PageRank score calculation")

pr = nk.centrality.PageRank(G)
pr.run()

out = open(path_out+"09_PageRank_ranking.txt", "w")
for e in pr.ranking() : 
	out.write(str(e)+"\n")
out.close()

logger.info("PageRank score calculation completed.")


#transformation to undirected graph
logger.info("Transformation to undirected graph")

G_undirected = nk.graphtools.toUndirected(G)
G_undirected.removeMultiEdges()

logger.info("Transformation to undirected graph ended.")



logger.info("clustering coefficient calculation")

lcc = nk.centrality.LocalClusteringCoefficient(G_undirected, turbo=True)
lcc.run()

out = open(path_out+"09_clusteringCoefficient_ranking.txt", "w")
for e in lcc.ranking() : 
	out.write(str(e)+"\n")
out.close()

logger.info("clustering coefficient calculation completed.")


logger.info("Community detection (Louvain)")

plmCommunities = nk.community.detectCommunities(G_undirected, algo=nk.community.PLM(G_undirected, True))
nk.community.writeCommunities(plmCommunities, path_out+"09_communitiesPLM.partition")

logger.info("Community detection (Louvain) completed.")


logger.info("KadabraBetweenness centrality calculation")

kadabra = nk.centrality.KadabraBetweenness(G_undirected, err=0.0001, delta=0.1)
kadabra.run()

out = open(path_out+"09_KadabraBetweenness_err0001_delta1_ranking.txt", "w")
for e in kadabra.ranking() : 
	out.write(str(e)+"\n")
out.close()

logger.info("KadabraBetweenness centrality calculation completed.")

