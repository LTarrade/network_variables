from collections import defaultdict
import numpy as np
import logging
import argparse
from numba import jit
from numba.typed import Dict
from numba.core import types
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("-n",  help="number of processes (must be the same for all processes)", type=int)
parser.add_argument("-m",  help="modulo of users to be processed (each process must have its own modulo; conditions the file name)", type=int)
parser.add_argument("-p",  help="path to working directory", type=str)
parser.add_argument("-c",  help="path to the community partition file", type=str)
parser.add_argument("-e",  help="Path to the file contains all the links between the users of the corpus", type=str)
args = parser.parse_args()

logging.basicConfig(
    filename=f"{args.p}randomWalk_{args.m}.log", 
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


logging.info('Loading network')
follows = defaultdict(list)
with open(args.e) as f:
    for line in f:
        try:
            user, follow = line.strip().split('\t')
            follows[int(user)].append(int(follow))
        except:
            continue
follows=dict(follows)
follows_numba = Dict.empty(key_type=types.int32, value_type=types.uint32[:])
for k,v in follows.items():
    follows_numba[k]=np.asarray(v, dtype='u4')

logging.info('network loaded, {} users'.format(len(follows)))

logging.info('Loading community')
community = {}
with open(args.c) as f:
    for i,l in enumerate(f):
        community[i] = int(l)
community_numba = Dict.empty(key_type=types.int64, value_type=types.int64)
for k,v in community.items():
    community_numba[k]=v
logging.info('communities loaded')


#@jit(nopython=True)
def treatNode(node,community=community,follows=follows,nWalks=10, maxLength=100):
    walks=[]
    for walk in range(nWalks):
        intialCommunity = community[node]
        walk=[node]
        while community[walk[-1]]==intialCommunity:
            if walk[-1] in follows:
                walk.append(np.random.choice(follows[walk[-1]]))
                if len(walk)==maxLength:
                    walk.append(np.inf)
                    break
            else:
                walk.append(np.nan)
                break
        walks.append(walk)
    return walks

res={}
logging.info('reading treated nodes')
with open(f'{args.p}randomWalks_{args.m}.pickle', 'rb') as f:
   res=pickle.load(f)
nodesToTreat = [k for k in community.keys() if k%args.n==args.m]
for i,node in enumerate(nodesToTreat):
    if node in res :
        continue
    res[node]=treatNode(node ,community=community_numba,follows=follows_numba,nWalks=10, maxLength=1000)
    if i%10000==0:
        logging.info(f'{i}/{len(nodesToTreat)} nodes treated')

logging.info(f'all nodes treated')
logging.info(f'pickleing...')
with open(f'{args.p}randomWalks_{args.m}.pickle','wb') as f:
    pickle.dump(res,f)
logging.info(f'Done.')
