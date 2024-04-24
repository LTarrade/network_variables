# network_variables

This repository contains the scripts used to calculate the network variables of the users in the corpus of tweets, to select the group of control words, to characterise the words, the scripts used to calculate the statistics relating to the differences between distributions and to visualise the distributions, as well as the code used for the predictions on buzz and changes.

1. network_var_networkit.py

	Script that, from the users' network and with the Networkit Python Library, retrieves for each node (user) its pageRank score, its clustering coefficient, its kadabraBetweenness centrality mesure, and the community it belongs to. 

	- **Input** : 

		- *--path_edges* : Path to the file contains all the ties between the users of the corpus. 
		The file should be in the following format:  <br/>
			\# Directed graph: \<name.ext\><br/>\# Nodes: \<nb_nodes\> Edges: \<nb_edges\><br/>\# source    target<br/>\<id_user\>  \<id_user\><br/>\<id_user\>  \<id_user\><br/>\<id_user\>  \<id_user\><br/>...<br/>\<id_user\>  \<id_user\> <br/>
			And source and target must be seprated by a tabulation. 

		- *--path_idUsers* : Path to json files containing the user IDs and corresponding users (for anonymisation).
		- *--path_out* : Path to the directory that will contain the output. 

	- **Output** : 

		- txt files containing the mesures associated to each node, and the partition file of communities

2. btw_InComm.py

	Script that, using the Networkit Python library, retrieves for each node (user) its measure of centrality in its community, then scales these measures by reducing the median to 0 and the IQR to 1 to make them comparable.

	- **Input** : 

		- *--path_edges* : Path to the file contains all the ties between the users of the corpus. 
		The file should be in the following format:  <br/>
			\# Directed graph: \<name.ext\><br/>\# Nodes: \<nb_nodes\> Edges: \<nb_edges\><br/>\# source    target<br/>\<id_user\>  \<id_user\><br/>\<id_user\>  \<id_user\><br/>\<id_user\>  \<id_user\><br/>...<br/>\<id_user\>  \<id_user\> <br/>
			And source and target must be seprated by a tabulation. 

		- *--path_idUsers* : Path to json files containing the user IDs and corresponding users (for anonymisation).
		- *--path_out* : Path to the directory that will contain the output. 
		- *--path_users* : Path to the dataframe of users, including a "LouvainCommunity_networkit" column containing the community associated with each user.

	- **Output** : 

		- the dataframe of users, with 3 new columns : 
			- "betweennessInCommunity", indicating the measure of centrality in the community associated with each user ;
			- "betweennessInCommunity_approx", indicating whether the measure of centrality is an exact or approximate measure ;
			- "betweennessInComm_toScale", indicating the measure of centrality in the community associated with each user scaling up so that measurements are comparable between all network users.

3. randomWalk.py

	Script that performs random walks from nodes in the network.

	- **Input** : 
		- *-c* : path to the community partition file
		- *-e* : path to the file contains all the links between the users of the corpus
	
	- **Output** :
		- files in pickle format containing in keys the identifiers of the nodes from which the random walks started, and in values as many lists of trajectories (nodes through which the walk passed before arriving at a node belonging to a community other than that of the starting node) as random walks carried out. 

4. selection_randomWordSamples.py

	Script that select 100 samples of 200 words with a relatively stable 5-year usage trajectory, each of which can act as a control group if matched with lexical innovations. 

	- **Input** : 

		- *--path_tokenizedTweets* : Path to the directory of tokenized tweets
		- *--path_out* : Path to the directory containing the output
		- *--path_users* : path to the dataframe containing the values of the network variables for each user
		- *--path_df* : path to the dataframe containing informations about the buzzes and changes
		- *--path_idByForm* : path to the json file containing the identifiers of words
		- *--path_idByUser* : path to the json file containing the identifiers of users
		- *--path_usersByMonth* : path to the json file containing the number of total users by month
		- *--path_occForms* : path to the json file containing the total number of occurrences of all forms by month 

	- **Output** : 

		- a dictionnary containing the number of occurrences and the users of all words with more than 100 occurrences
		- a file containing the list of all words that have little variance over the selected period
		- a dictionnary containing the total number of users of this words 
		- 100 dataframes, each containing 200 of these words, whose distribution in terms of number of users is similar to that of the lexical innovations 

5. medByForm.py

	Script retrieving for each innovation (buzz and change), and for each phase of diffusion, the users who adopt it at that phase; then retrieving for each network variable the median value of each of these sets to characterise each of the words. For the control words, the same procedure is used except that all their users over the 5 years considered are taken into account.  

	- **Input** :

		- *--path_out* : Path to the directory containing the output
		- *--path_users* : path to the dataframe containing the values of the network variables for each user
		- *--path_df* : path to the dataframe containing informations about the buzzes and changes
		- *--path_idByForm* : path to the json file containing the identifiers of words
		- *--path_idByUser* : path to the json file containing the identifiers of users
		- *--path_usersByMonthByForm* : path to the json file containing the number of total users by month by form
		- *--path_randomWords* : path to the directory containing the samples of random words (dataframes)
		- *--path_randomWordsUsers* : path to the json file containing the users of random words by month

	- **Output** : 

		- as many dataframes as there are sample randoms, each containing for each control word, each buzz, and each change, the median values of their users at each phase, for each of the network variables considered.

6. distribAndStats_and_prediction.py

	Script allowing to visualize the distributions of buzz, change, control words at different phases and according to each of the variables; to calculate the corresponding statistics, and to make predictions about the fate of lexical innovations in the innovation and propagation phases.  

	- **Input** :

		- *--path_out* : path to the directory containing the output
		- *--path_df* : path to the dataframe containing median values by word and by period
		- *--path_pred* : path to the dictionary that contains the prediction results if it already exists
	

	- **Output** : 

		- results of non-parametric tests on the pairs of distributions
		- visualisations of distributions of changes, buzz, and control words at the different phases and for each network variable 
		- prediction results in terms of accuracy, AUC score, and confusion matrices
		- visualisations of the distributions of results in terms of accuracy and AUC score for prediction

7. Data 
	- **df_varNetwork_medByForm.csv** : contains the median value associated with each word (buzzes, changes, controls) for each variable and each phase. 
	- **clients_list.txt** : contains the list of Twitter clients for whom tweets have been kept.
