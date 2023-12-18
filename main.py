#Import necessary libraries

import networkx as nx
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
from networkx.algorithms.community import greedy_modularity_communities
import argparse
import os
print("start", flush=True)

#The function reads a graph from a CSV file and returns a NetworkX graph.
def read_csv_as_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        row1 = next(reader) 
        for row in reader:
            source, target = row
            G.add_edge(source, target)
    return G

#The function reads a graph from a file such as with extentions ".edges" and returns a NetworkX graph.
def read_file_as_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r', encoding="utf8") as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split()  # Split line by whitespace (tabs or spaces)
            if len(parts) >= 2:
                source, target = parts[:2]
                G.add_edge(source, target)
    return G

#Prints and returns various statistics of the input graph G.
def stats_of_dataset(G):
    print("Number of edges",G.number_of_edges())
    print("Number of nodes",G.number_of_nodes())
    print("Average Degree of the graph",sum(dict(G.degree()).values())/G.number_of_nodes())
    print("Average clustering coeffient",nx.average_clustering(G))
    # Detect communities before sampling
    communities_before_sampling = list(greedy_modularity_communities(G))
    print("Communities:", communities_before_sampling)
    return G.number_of_edges(),G.number_of_nodes(),sum(dict(G.degree()).values())/G.number_of_nodes(),nx.average_clustering(G),list(greedy_modularity_communities(G))
    
def check_isomorphic(G1,G2):
    # Check if the neighborhoods are isomorphic
    return nx.is_isomorphic(G1, G2)
        
def precompute_neighborhoods(G):
    # Precompute 1-hop neighborhoods for all nodes
    neighborhoods = {}
    for node in G.nodes():
        neighborhoods[node] = nx.ego_graph(G, node, radius=1)
    return neighborhoods

def occurrence_frequency(G, check_node, neighborhoods):
    # Retrieve precomputed neighborhood for the check_node
    neighborhood = neighborhoods[check_node]
    count = 0
    for node in G.nodes():
        # Retrieve precomputed neighborhood for the current node
        neighborhood_item = neighborhoods[node]
        if check_isomorphic(neighborhood, neighborhood_item):
            count += 1
    return count

# Computes the uniqueness of neighborhoods in the input graph.
def uniqueness_of_neighborhoods(graph):
    nodes = graph.nodes()
    neighborhoods = precompute_neighborhoods(graph)
    value = 0
    for i in tqdm(nodes, desc="Checking uniqueness"):
        if(occurrence_frequency(graph,i,neighborhoods)==1):
            value+= 1
    uniqueness = value/len(graph.nodes())
    return uniqueness

#Sample edges randomly (Remove edges randomly using sampling rate)
def sample_edges_uniformly(graph, sampling_rate):

    edges = list(graph.edges())
    num_edges_to_remove = len(edges) - int(len(edges) * sampling_rate)
    edges_to_retain = random.sample(edges, num_edges_to_remove)

    sampled_graph = nx.Graph()
    sampled_graph.add_nodes_from(graph)
    sampled_graph.add_edges_from(edges_to_retain)
    return sampled_graph

#Sample edges Biased (Remove edges biased using sampling rate)
def biased_edge_sampling(graph, sampling_rate):
    edges = list(graph.edges())
    degrees = dict(graph.degree())

    # Calculate the probability of selecting each edge based on the degree of nodes
    edge_probabilities = [degrees[edge[0]] * degrees[edge[1]] for edge in edges]
    total_sum = sum(edge_probabilities)
    normalized_probabilities = [p / total_sum for p in edge_probabilities]

    # Sample edges based on these probabilities
    sampled_edges = random.choices(edges, weights=normalized_probabilities, k=int(len(edges) * sampling_rate))
    graph.remove_edges_from(sampled_edges)

    return graph

#Sample edges based on unique neighborhoods (Remove edges randomly of unique neighborhoods using sampling rate)
def check_for_unique_neighborhoods(graph,sampling_rate):
    #If it is isomorphic neighborhoods leave them
    #The unique neighborhoods check them and remove edges from them
    number_of_edges = int(sampling_rate*graph.number_of_edges())
    edges_removed = 0
    
    neighborhoods = precompute_neighborhoods(graph)    
    for node in tqdm(graph.nodes()):
        # Check if any other node has the same neighborhood
        for other_node in graph.nodes():     
            while node != other_node and (check_isomorphic(neighborhoods[other_node], neighborhoods[node])==False) and edges_removed<number_of_edges:
                edges_of_node = list(graph.edges(node))
                edges_to_remove = random.sample(edges_of_node, int(len(edges_of_node) *sampling_rate))
                graph.remove_edges_from(edges_to_remove)  
                edges_removed+= len(edges_to_remove)
                    # No need to continue checking
                break
    return graph

#Remove edges for nodes that have a unique degree 
def check_unique_degree(graph):  
    unique_degree_nodes = []
    avg_degree = sum(dict(graph.degree()).values())/graph.number_of_nodes()
    #print(avg_degree)
    degrees = [val for (node, val) in graph.degree()]
    values = [[x,degrees.count(x)] for x in set(degrees)]
    for i in values:
        if(i[1]==1):
            if(i[0] > avg_degree):
                #find node that has this degree
                node = [node for node, degree in graph.degree() if degree == i[0]]
                unique_degree_nodes.append(node[0])
    return unique_degree_nodes

def remove_edges_with_unique_degree(graph,sampling_rate):
    unique_degrees_nodes =  check_unique_degree(graph)
    number_of_edges = int(sampling_rate*graph.number_of_edges())
    edges_removed = 0
    avg_degree = sum(dict(graph.degree()).values())/graph.number_of_nodes()
    for node in unique_degrees_nodes:
        previous_degree = graph.degree(node)
        new_degree = 0
        while graph.degree(node) >= avg_degree and previous_degree!=new_degree and edges_removed<=number_of_edges:
            previous_degree = graph.degree(node)
            edges_of_node = list(graph.edges(node))
            edges_to_remove = random.sample(edges_of_node, int(len(edges_of_node) * sampling_rate))
            graph.remove_edges_from(edges_to_remove)  
            new_degree = graph.degree(node)
            edges_removed+= len(edges_to_remove)
    return graph


#MAIN CODE
RED = "\033[31m"

# Define command-line arguments that can be taken
argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset",  type=str, help="Location of the dataset")
argparser.add_argument("--sampling_rate", default=0.1, type=float, help="Value of the sampling rate")
argparser.add_argument("--seed", default=1, type=int, help="Seed for reporducibility")
args = argparser.parse_args()

# Set the seed for reproducibility
np.random.seed(args.seed)

# Create an empty DataFrame to store results
data = pd.DataFrame(columns=["Graph","Number of edges","Number of nodes","Average Degree","Average clustering coeffient",
                            "Communities Structure","Uniqueness"])

# Read the graph from the specified dataset
if str(args.dataset).endswith(".csv"):
    G = read_csv_as_graph(args.dataset)
else:
    G = read_file_as_graph(args.dataset)  
    
#Calculate statistics of the dataset
edges,nodes,avg_degree,average_clustering,communities = stats_of_dataset(G)
uniqueness = str(uniqueness_of_neighborhoods(G))
print(uniqueness)

# Create a new row with the computed statistics
new_row = {"Graph":"Original Graph","Number of edges":edges,"Number of nodes":nodes,"Average Degree":avg_degree,"Average clustering coeffient":average_clustering,
                            "Communities Structure":communities,"Uniqueness":uniqueness}

# Append the new row to the DataFrame
data.loc[len(data)] = new_row

# print(RED + "Default: " + str(uniqueness_of_neighborhoods(G)))

G_edges_uniformly = sample_edges_uniformly(G, args.sampling_rate)
edges,nodes,avg_degree,average_clustering,communities = stats_of_dataset(G_edges_uniformly)
uniqueness = str(uniqueness_of_neighborhoods(G_edges_uniformly))
print(uniqueness)

new_row = {"Graph":"Random Sampling","Number of edges":edges,"Number of nodes":nodes,"Average Degree":avg_degree,"Average clustering coeffient":average_clustering,
                            "Communities Structure":communities,"Uniqueness":uniqueness}

# Append the new row to the DataFrame
data.loc[len(data)] = new_row
# print(RED + "Edges uniformly: " + str(uniqueness_of_neighborhoods(G_edges_uniformly)))

G_biased_edge_sampling = biased_edge_sampling(G, args.sampling_rate)
edges,nodes,avg_degree,average_clustering,communities = stats_of_dataset(G_biased_edge_sampling)
uniqueness = str(uniqueness_of_neighborhoods(G_biased_edge_sampling))
print(uniqueness)

new_row = {"Graph":"Biased Edge Sampling","Number of edges":edges,"Number of nodes":nodes,"Average Degree":avg_degree,"Average clustering coeffient":average_clustering,
                            "Communities Structure":communities,"Uniqueness":uniqueness }

# Append the new row to the DataFrame
data.loc[len(data)] = new_row
# print(RED + "Biased edge sampling: " + str(uniqueness_of_neighborhoods(G_biased_edge_sampling)))

G_remove_unique_degree = remove_edges_with_unique_degree(G, args.sampling_rate)
edges,nodes,avg_degree,average_clustering,communities = stats_of_dataset(G_remove_unique_degree)
uniqueness = str(uniqueness_of_neighborhoods(G_remove_unique_degree))
print(uniqueness)

new_row = {"Graph":"Sample edges that have an unique degree","Number of edges":edges,"Number of nodes":nodes,"Average Degree":avg_degree,"Average clustering coeffient":average_clustering,
                            "Communities Structure":communities,"Uniqueness":uniqueness}

# Append the new row to the DataFrame
data.loc[len(data)] = new_row
# print(RED + "Unique degree sampling: " + str(uniqueness_of_neighborhoods(G_remove_unique_degree)))

G_remove_traingles = check_for_unique_neighborhoods(G, args.sampling_rate)
edges,nodes,avg_degree,average_clustering,communities = stats_of_dataset(G_remove_traingles)
uniqueness = str(uniqueness_of_neighborhoods(G_remove_traingles))
print(uniqueness)

new_row = {"Graph":"Sample edges that have an unique neighborhood","Number of edges":edges,"Number of nodes":nodes,"Average Degree":avg_degree,"Average clustering coeffient":average_clustering,
                            "Communities Structure":communities,"Uniqueness":uniqueness}

# Append the new row to the DataFrame
data.loc[len(data)] = new_row


RDIR = "./results/"
# check if results directory exists, if not, create it
if not os.path.isdir(RDIR):
    os.mkdir(RDIR)
file_name = RDIR+"d"+ str(args.dataset.split("\\")[1])+"-sr"+str(args.sampling_rate)+"-s"+str(args.seed)+".csv"

print(f"Storing results in {RDIR}")
data.to_csv(file_name)