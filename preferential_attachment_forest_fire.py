import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import csv
import numpy
import itertools
from collections import deque
from scipy.stats import linregress, geom, uniform
from math import ceil, log
import cProfile
import profile
from networkx.utils import pareto_sequence, zipf_sequence


# Import node labels from file into edge_list
# to create start graph
def import_graph(start_with_n_edges):
    fileFirst100 = 'GraphEdgesFirst100.csv'
    fileFirst1000 = 'GraphEdgesFirst1000.csv'
    Fileall = 'GraphEdgesAll.csv'
    Filea929394 = 'GraphEdges19929394.csv'
    Filea9293 = 'GraphEdges199293.csv'
    operator = 'r'
    graph_edges = csv.reader(open(Filea9293, operator))
    edge_list = []
    for edge in graph_edges:
    #for edge in itertools.islice(graph_edges, start_with_n_edges):
        edges = ((edge[0]), (edge[1]))
        edge_list.append(edges)
    return edge_list


# Export graph stats to csv file
def export_graph_stats(stats):
    FileOut = 'Graph_Stats.csv'
    operator = 'a'
    outfile  = open(FileOut, operator)
    graph_stats = csv.writer(outfile, delimiter = ',')
    for row in stats:
        graph_stats.writerow(row)
    outfile.close()


def export_graph_degree_sequence(sequence):
    FileOut = 'Graph_Degree_Sequence.csv'
    operator = 'a'
    outfile  = open(FileOut, operator)
    graph_degree_sequence = csv.writer(outfile, delimiter = ',')
    degree_sequence = sorted(sequence, reverse = True)
    graph_degree_sequence.writerow(degree_sequence)
    outfile.close()
    

def export_graph_in_degree_sequence(sequence):
    FileOut = 'Graph_In_Degree_Sequence.csv'
    operator = 'a'
    outfile  = open(FileOut, operator)
    graph_degree_sequence = csv.writer(outfile, delimiter = ',')
    degree_sequence = sorted(sequence, reverse = True)
    graph_degree_sequence.writerow(degree_sequence)
    outfile.close()


def export_graph_out_degree_sequence(sequence):
    FileOut = 'Graph_Out_Degree_Sequence.csv'
    operator = 'a'
    outfile  = open(FileOut, operator)
    graph_degree_sequence = csv.writer(outfile, delimiter = ',')
    degree_sequence = sorted(sequence, reverse = True)
    graph_degree_sequence.writerow(degree_sequence)
    outfile.close()


# Export graph edges to csv file
def export_graph(stats):
    FileOut = 'GraphOutput.csv'
    operator = 'w'
    outfile  = open(FileOut, operator)
    graph_edges = csv.writer(outfile, delimiter = ',')
    for edge in g.edges_iter():
        graph_edges.writerow(edge)
    outfile.close()

    
# Transforms node labels to node indexes
# if first three edge labels are eg: (50,3), (11,7), (12,11)
# First three edge indexes will be (1,0), (3,2), (4,3)
def generate_graph(start_with_n_edges):
    g = nx.DiGraph()
    edge_list = import_graph(start_with_n_edges)
    node_list = combine_list(edge_list)
    unique_nodes = unique_list(node_list)
    new_nodes = (range(len(unique_nodes)))
    edge_index = map_list(edge_list, unique_nodes, new_nodes)
    g.add_edges_from(edge_index)
    return g


# This searches the orginal graph and returns
# nodes for each node in a given community
def repeated_items(repeated_items, subset):
    node_list = []
    for nodes in subset:
        for edges in repeated_items:
            if nodes == edges:
                node_list.append(nodes)
    return node_list


# Splits a list of lists into one list
# eg: [(1,0),(2,1)] becomes [1,0,2,1]
# We want to keep duplicates here
def combine_list(lists):
    return [item for sublist in lists for item in sublist]


# This maps item1, item2 in one list to items in
# another list and maps items in the second list
# to items in a third list and returns the mapping
# in third list for item1, item2
def map_list(list_of_lists, list1, list2):
    temp_list = []
    temp_list2 = []
    for u,v in list_of_lists:
        for w,x in zip(list1, list2):
            if u == w:
                temp_list.append(x)
            if v == w:
                temp_list2.append(x)
    return zip(temp_list, temp_list2)


# Removes duplicates from a list of items
def unique_list(list_items):
   keys = {}
   for e in list_items:
       keys[e] = 1
   return keys.keys()


# Chooses a random subset of n unique items
# from a list containing duplicates
# random.sample can sometimes return duplicates
# resulting in the return of less than n items
def random_subset(List_of_choices, n_items):
    choices = []
    while len(choices) < n_items:
        selection = random.choice(List_of_choices)
        if selection not in choices:
            choices.append(selection)
    return choices


# Ads n new nodes to the graph at each time step
def add_n_nodes_per_step(g, nodes_per_step):
    this_node = g.number_of_nodes() + 1
    nodes_this_step = range(this_node, this_node + nodes_per_step)
    g.add_nodes_from(nodes_this_step)
    return nodes_this_step


# Add complete list of edges at each time step
# between new nodes and selected existing nodes
def add_edges_per_step(g, this_node, edges_this_step):
    edges_per_step = len(edges_this_step)
    add_edges_this_step = zip([this_node] * edges_per_step, edges_this_step)
    g.add_edges_from(add_edges_this_step)
    return add_edges_this_step

        
# Generate geometrically distributed random number
def generate_geometric(p):
    #return geom.rvs(1 - p)
    if (p >= 1 or p == 0):
        return 0
    else:
        r = 1.0 - random.random() # never zero
        geometric = int(ceil(log(r) / log(1.0 - p)))
        return geometric


# Generate random number form pareto distribution
# with given exponent
def generate_pareto(exponent):
    return pareto_sequence(1, exponent)


# Generate random number form zipf distribution
# with given exponent
def generate_zipf(exponent):
    return zipf_sequence(1, exponent, 1)


# Generate uniform random variable with max value a
def generate_uniform(a):
    return random.randint(1, a)
    

def pref_attach_draw(g, repeated_nodes, edges_per_step):
    nodes_last_step = repeated_nodes
    if edges_per_step > g.number_of_nodes():
        n_edges_per_step = g.number_of_nodes()
    else:
        n_edges_per_step = edges_per_step
    nodes_this_step = random_subset(nodes_last_step, n_edges_per_step)
    return nodes_this_step


def pref_attach_roulette(subset, edges_per_step):
    pop_size = len(subset)
    if edges_per_step >= pop_size:
        n_edges_per_step = pop_size
    else:
        n_edges_per_step = edges_per_step
    total_fitness = float(sum(subset.values()))
    rel_fitness = [f / total_fitness for f in subset.values()]
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    nodes_this_step = set() 
    while len(nodes_this_step) < n_edges_per_step:
        r = random.random()
        for (i, node) in enumerate(subset.keys()):
            if r <= probs[i]:
                nodes_this_step.add(node)
                break
    return nodes_this_step


def forest_fire_attach(g, edges_in_iter_step, visited_nodes,
                       this_child, decay, backward_burning_ratio,
                       forward_burning_probability):
    in_links_this_step = pref_attach_roulette(
                    g.degree(g.predecessors(this_child)),
                    generate_geometric(
                    (backward_burning_ratio) + decay))
    out_links_this_step = pref_attach_roulette(
                    g.degree(g.successors(this_child)),
                    generate_geometric(
                    (forward_burning_probability) + decay))
    edges_in_iter_step.extend(in_links_this_step)
    edges_in_iter_step.extend(out_links_this_step)
    visited_nodes.add(this_child)
    return edges_in_iter_step

                
def forest_fire(g, ambassadors_this_step, edges_this_step,
                backward_burning_ratio, forward_burning_probability,
                decay_rate):
    decay = 0
    queue = deque([(iter([ambassadors_this_step]))])
    visited_nodes = set()
    while queue and decay < 1.0:               
        children = queue[0]
        edges_in_iter_step = []
        try:
            child = next(children)
            for this_child in (child):
                if this_child not in visited_nodes:
                    edges_in_iter_step = forest_fire_attach(g,
                                    edges_in_iter_step, visited_nodes,
                                    this_child, decay, backward_burning_ratio,
                                    forward_burning_probability)
                edges_this_step.extend(set(edges_in_iter_step))
            if not len(edges_in_iter_step):
                break
            else:
                queue.append((iter([set(edges_in_iter_step)])))
                decay += random.random()
        except StopIteration:
            queue.popleft()                          
    return set(edges_this_step)
                

def emulate_graph(run, time_steps, print_stats_every_n_steps, nodes_per_step,
                  max_ambassadors_per_step, backward_burning_ratio,
                  forward_burning_probability, decay_rate, start_with_n_edges):
    g = generate_graph(start_with_n_edges)
    t = 1
    stats = []
    while t <= time_steps:
        nodes_this_step = add_n_nodes_per_step(g, nodes_per_step)
        for this_node in nodes_this_step:
            edges_this_step = []
            ambassadors_this_step = pref_attach_roulette(
                            g.degree(g.nodes()),
                            generate_uniform(max_ambassadors_per_step))
            edges_this_step.extend(ambassadors_this_step)
            links_this_step = forest_fire(g, ambassadors_this_step, edges_this_step,
                            backward_burning_ratio, forward_burning_probability,
                            decay_rate)
            edges_this_step.extend(links_this_step)
            add_edges_this_step = add_edges_per_step(g, this_node, edges_this_step)
        if t % print_stats_every_n_steps == 0:
            graph_stats(run, t, g, stats, max_ambassadors_per_step,
                        forward_burning_probability, backward_burning_ratio)
        t += 1
    #draw_graph(g)
    #plot_degree_histogram(g)
    #plot_in_degree_histogram(g)
    #plot_out_degree_histogram(g)
    #export_graph(g)
    export_graph_stats(stats)
    export_graph_degree_sequence(g.degree().values())
    export_graph_in_degree_sequence(g.in_degree().values())
    export_graph_out_degree_sequence(g.out_degree().values())
    

# Returns the largest connected component of the graph
# The graph is converted to an undirected graph for this
# Should be using strongest_connected_component_subgraphs
# for directed graph here but having issues computing a diameter
def connected_component_subgraphs(g):
    cc_copy = g.copy().to_undirected()
    return max(sorted(nx.connected_component_subgraphs(cc_copy),
              reverse = True), key = len)
    #return max(sorted(nx.strongly_connected_component_subgraphs(g),
              #reverse = True), key = len)


# Returns the diameter of the largest connected component of the graph
def graph_diameter(g):
    cc = connected_component_subgraphs(g)
    return nx.diameter(cc)


# Returns the diameter of the largest connected component of the graph
def clustering_coefficient(g):
    cc = connected_component_subgraphs(g)
    return nx.average_clustering(cc)


# Returns the diameter of the largest connected component of the graph
def graph_diameter_and_clustering(g):
    cc = connected_component_subgraphs(g)
    return nx.diameter(cc), nx.average_clustering(cc)


'''
# Plots the degree rank plot for the graph
def plot_degree_histogram(g):
    degree_sequence = sorted(g.degree().values(), reverse = True)
    dmax = max(degree_sequence)
    plt.figure()
    plt.loglog(degree_sequence, 'b-', marker = 'o')
    plt.title("Degree rank plot")
    plt.xlabel("rank")
    plt.ylabel("degree")
    plt.savefig("degree_histogram.png")
    #plt.show()

    
# Plots the in-degree rank plot for the graph
def plot_in_degree_histogram(g):
    degree_sequence = sorted(g.in_degree().values(), reverse = True)
    dmax = max(degree_sequence)
    plt.figure()
    plt.loglog(degree_sequence, 'b-', marker = 'o')
    plt.title("In-Degree rank plot")
    plt.xlabel("rank")
    plt.ylabel("degree")
    plt.savefig("in_degree_histogram.png")
    #plt.show()


# Plots the out-degree rank plot for the graph
def plot_out_degree_histogram(g):
    degree_sequence = sorted(g.out_degree().values(), reverse = True)
    dmax = max(degree_sequence)
    plt.figure()
    plt.loglog(degree_sequence, 'b-', marker = 'o')
    plt.title("Out-Degree rank plot")
    plt.xlabel("rank")
    plt.ylabel("degree")
    plt.savefig("out_degree_histogram.png")
    #plt.show()


# Draws the final graph
def draw_graph(g, labels = None, graph_layout = 'random',
           node_size = 400, node_color = 'red', node_alpha = 0.5,
           node_text_size = 8,
           edge_color='black', edge_alpha = 0.3, edge_tickness = 1,
           edge_text_pos = 8,
           text_font = 'sans-serif'):
    if graph_layout == 'circular':
        graph_pos = nx.circular_layout(g)
    elif graph_layout == 'spring':
        graph_pos = nx.spring_layout(g)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(g)
    else:
        graph_pos = nx.random_layout(g)
    nx.draw_networkx_nodes(g, graph_pos, node_size = node_size, 
               alpha = node_alpha, node_color = node_color)
    nx.draw_networkx_edges(g, graph_pos, width = edge_tickness,
               alpha = edge_alpha, edge_color = edge_color)
    nx.draw_networkx_labels(g, graph_pos, font_size = node_text_size,
                font_family = text_font)
    plt.axis('off')
    plt.title("Graph Plot")
    plt.savefig('graph_plot.png')
    #plt.show()
'''

# Prints the below stats at each nth time steambassadors_per_stepp
def print_stats(stats):
    print [stats[0],stats[1],stats[2],stats[3],stats[4],stats[5],stats[6],
          round(stats[7],4),round(stats[8],2),stats[9],round(stats[10],2),
          stats[11],round(stats[12],4),stats[13],round(stats[14],4),
          stats[15],round(stats[16],4)]


# Collects stats at each nth time step and appends them
# as a list to a list of lists for plotting diameter
# and ratio of edges to nodes. Also calls print_stats
def graph_stats(run, t, g, stats, a, p, r):
    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    density = nx.density(g)
    ratio = edges / float(nodes)
    diameter, clustering = graph_diameter_and_clustering(g)
    
    degree_sequence = sorted(g.degree().values(), reverse = True)
    in_degree_sequence = sorted(g.in_degree().values(), reverse = True)
    out_degree_sequence = sorted(g.out_degree().values(), reverse = True)
    
    degreemax = max(degree_sequence)
    slope, intercept, r_value, p_value, std_err = linregress(
                    range(len(degree_sequence)), degree_sequence)
    degreeslope = slope

    indegreemax = max(in_degree_sequence) 
    in_slope, in_intercept, in_r_value, in_p_value, in_std_err = linregress(
                    range(len(in_degree_sequence)), in_degree_sequence)
    indegreeslope = in_slope
    
    outdegreemax = max(out_degree_sequence)
    out_slope, out_intercept, out_r_value, out_p_value, out_std_err = linregress(
                    range(len(out_degree_sequence)), out_degree_sequence)
    outdegreeslope = out_slope
    
    stats_this_step = [run, t, a, p, r, nodes, edges, density, ratio, diameter,
                       clustering, degreemax, degreeslope, indegreemax,
                       indegreeslope, outdegreemax, outdegreeslope]
    
    stats.append(stats_this_step)
    print_stats(stats_this_step)

            
def main():

    start_with_n_edges = 0 # Not used
    # Number of time steps
    time_steps = 4000
    # Number of nodes to add at each time step   
    nodes_per_step = 1
    # Number of edges to add at each time step
    # max_ambassadors_per_step = 1
    # Collect and print stats at each nth time step
    print_stats_every_n_steps = 100
    # forward_burning_probability = 0
    # backward_burning_ratio = 0
    decay_rate = random.random()
    
    create_n_link_params = [(0.0,0.0),
                            (0.4,0.6),
                            (0.4,0.8),
                            (0.5,0.5),
                            (0.6,0.4),
                            (0.6,0.8),
                            (0.8,0.4),
                            (0.8,0.6),
                            (0.9,0.7),
                            (0.9,0.8)]

    max_ambassadors_params = [1,2,3,4]
            
    run = 1     
    for max_ambassadors_per_step in max_ambassadors_params:
        for forward_burning_probability, backward_burning_ratio in create_n_link_params:        
            emulate_graph(run, time_steps,
                        print_stats_every_n_steps,
                        nodes_per_step,
                        max_ambassadors_per_step,
                        backward_burning_ratio,
                        forward_burning_probability,
                        decay_rate,
                        start_with_n_edges)
            run += 1


if __name__ == "__main__":
    main()