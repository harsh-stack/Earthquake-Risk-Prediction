# %%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# %%
# Load the data
df_real = pd.read_csv('D:/SWN Projects/trade_1988_2021.csv')

# %%
df = df_real[df_real['PartnerName'] != 'World']

# %%
# Create a new column for the interval (5-year intervals)
df['Interval'] = (df['Year'] // 4) * 4

# %%
# Create a list to store the datasets for each interval
datasets_by_interval = []
trade_value_per_interval = []
radius_per_interval = []
diameter_per_interval = []
avg_path_length_per_interval = []
avg_clustering_per_interval = []

# %%
# Split the original DataFrame into separate DataFrames for each interval and store them in the list
for interval in sorted(df['Interval'].unique()):
    interval_df = df[df['Interval'] == interval]
    datasets_by_interval.append((interval, interval_df))

# Initialize a list to store the number of edges for each interval
edges_per_interval = []

# %%
# Function to process each dataset and visualize the graph
def process_and_visualize_graph(interval, df):
    G = nx.Graph()

    for _, row in df.iterrows():
        if G.has_edge(row['ReporterName'], row['PartnerName']):
            G[row['ReporterName']][row['PartnerName']]['TradeValue'] += row['TradeValue']
        else:
            G.add_edge(row['ReporterName'], row['PartnerName'], TradeValue=row['TradeValue'])

    node_sizes = []
    for node in G.nodes():
        total_weight = sum(G[node][neighbor]['TradeValue'] for neighbor in G.neighbors(node))
        node_sizes.append(total_weight)

    max_node_size = max(node_sizes)
    node_sizes = [size / max_node_size * 10 for size in node_sizes]

    pos = nx.spiral_layout(G) 
    node_sizes = [9 * G.degree(node) for node in G.nodes]

    # Calculate the node with the maximum and minimum connections
    degree_dict = dict(G.degree())
    max_connect_count = max(degree_dict, key=degree_dict.get)
    min_connect_count = min(degree_dict, key=degree_dict.get)

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.4)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="black")
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_nodes(G, pos, nodelist=[max_connect_count], node_color="red", label=f"Highest: {max_connect_count}")
    nx.draw_networkx_nodes(G, pos, nodelist=[min_connect_count], node_color="green", label=f"Lowest: {min_connect_count}")

    labels = {node: node for node in G.nodes if G.degree(node) > 20}  
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    plt.title(f"Trade Network Visualization for {interval} - {interval + 3} with Node Sizes Based on Weights", fontsize=10)
    plt.show()
    
    # Clustering Coefficient Analysis
    cc = nx.clustering(G)  
    avg_cc = nx.average_clustering(G)  

    print(f"\nAverage Clustering Coefficient of the Network in {interval} - {interval + 3}:")
    print(avg_cc)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(num_nodes, num_edges)

    average_degree = np.mean([degree for node, degree in G.degree()]) 
    print(f"Average Degree: {average_degree:.4f}")

    average_clustering_coefficient = nx.average_clustering(G) 
    print(f"Average Clustering Coefficient: {average_clustering_coefficient:.4f}")

    average_path_length = nx.average_shortest_path_length(G)
    print(f"Average Path Length: {average_path_length:.4f}")

    # Using Erdős-Rényi model with the same number of nodes and probability to match the average degree 
    p = average_degree / (num_nodes - 1) 
    random_graph = nx.erdos_renyi_graph(num_nodes, p)

    print(f"Random Graph Number of Nodes: {random_graph.number_of_nodes()}") 
    print(f"Random Graph Number of Edges: {random_graph.number_of_edges()}")

    random_clustering_coefficient = nx.average_clustering(random_graph) 
    print(f"Random Graph Average Clustering Coefficient: {random_clustering_coefficient:.4f}")

    random_path_length = nx.average_shortest_path_length(random_graph) 
    print(f"Random Graph Average Path Length: {random_path_length:.4f}")

    print(f"Avg. Path Length - SWN: {average_path_length:.4f}, Random: {random_path_length:.4f}") 
    print(f"Avg. Clustering Coeff. - SWN: {average_clustering_coefficient:.4f}, Random: {random_clustering_coefficient:.4f}")
    
    # Append the number of edges to the list
    edges_per_interval.append((interval, num_edges))
    total_trade_value = df['TradeValue'].sum()
    trade_value_per_interval.append((interval, total_trade_value))
    radius = nx.radius(G)
    radius_per_interval.append((interval, radius))
    diameter = nx.diameter(G)
    diameter_per_interval.append((interval, diameter))
    avg_path_length_per_interval.append((interval, average_path_length))
    avg_clustering_per_interval.append((interval, average_clustering_coefficient))

# %%
import warnings
warnings.filterwarnings('ignore')

def assign_interval(year):
    intervals = [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
    for i, interval_start in enumerate(intervals):
        if i == len(intervals) - 1:
            return interval_start
        elif year >= interval_start and year < intervals[i + 1]:
            return interval_start
    return intervals[-1]

def create_robust_trade_network(df_real, interval, min_trade_threshold=1000):
    interval_data = df_real[df_real['Interval'] == interval]
    interval_data = interval_data[(interval_data['ReporterName'] != interval_data['PartnerName'])]
    interval_data = interval_data[~interval_data['ReporterName'].isin(['World', 'Areas, nes', 'Other Asia, nes'])]
    interval_data = interval_data[~interval_data['PartnerName'].isin(['World', 'Areas, nes', 'Other Asia, nes'])]
    aggregated = interval_data.groupby(['ReporterName', 'PartnerName'])['TradeValue'].sum().reset_index()
    aggregated = aggregated[aggregated['TradeValue'] >= min_trade_threshold]
    G = nx.Graph()
    for _, row in aggregated.iterrows():
        reporter = row['ReporterName']
        partner = row['PartnerName']
        weight = row['TradeValue']
        G.add_edge(reporter, partner, weight=weight)
    return G

def calculate_omega_coefficient(G, niter=1000):
    if len(G.nodes()) < 4 or len(G.edges()) < 2:
        return None
    C_obs = nx.average_clustering(G)
    if nx.is_connected(G):
        L_obs = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc)
        L_obs = nx.average_shortest_path_length(G_cc)
    n = len(G.nodes())
    k = int(2 * len(G.edges()) / n)
    C_latt_values = []
    for _ in range(niter):
        try:
            G_latt = nx.watts_strogatz_graph(n, k, 0)
            C_latt_values.append(nx.average_clustering(G_latt))
        except:
            continue
    degree_sequence = [d for n, d in G.degree()]
    L_rand_values = []
    for _ in range(niter):
        try:
            G_rand = nx.configuration_model(degree_sequence)
            G_rand = nx.Graph(G_rand)
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
            if nx.is_connected(G_rand):
                L_rand_values.append(nx.average_shortest_path_length(G_rand))
        except:
            continue
    if not C_latt_values or not L_rand_values:
        return None
    C_latt = np.mean(C_latt_values)
    L_rand = np.mean(L_rand_values)
    omega = (L_rand / L_obs) - (C_obs / C_latt)
    classification = classify_small_world_omega(omega)
    return {
        'omega': omega,
        'C_obs': C_obs,
        'C_latt': C_latt,
        'L_obs': L_obs,
        'L_rand': L_rand,
        'classification': classification
    }

def classify_small_world_omega(omega):
    if -0.5 <= omega <= 0.5:
        return "Small-world"
    elif omega > 0.5:
        return "Random-like"
    else:
        return "Lattice-like"

def statistical_significance_test(observed_omega, null_omegas, alpha=0.05):
    if len(null_omegas) < 30:
        return False, 1.0
    p_value = 2 * min(np.mean(null_omegas <= observed_omega), np.mean(null_omegas >= observed_omega))
    return p_value < alpha, p_value

def comprehensive_small_world_analysis(df_real):
    df_real['Interval'] = df_real['Year'].apply(assign_interval)
    intervals = [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
    results = []
    for interval in intervals:
        G = create_robust_trade_network(df_real, interval)
        if len(G.nodes()) < 10 or len(G.edges()) < 20:
            continue
        sw_metrics = calculate_omega_coefficient(G, niter=100)
        if sw_metrics:
            null_omegas = []
            degree_sequence = [d for n, d in G.degree()]
            for _ in range(100):
                try:
                    G_null = nx.configuration_model(degree_sequence)
                    G_null = nx.Graph(G_null)
                    G_null.remove_edges_from(nx.selfloop_edges(G_null))
                    null_metrics = calculate_omega_coefficient(G_null, niter=20)
                    if null_metrics:
                        null_omegas.append(null_metrics['omega'])
                except:
                    continue
            is_significant, p_value = statistical_significance_test(sw_metrics['omega'], null_omegas)
            result = {
                'interval': interval,
                'n_countries': len(G.nodes()),
                'n_edges': len(G.edges()),
                'omega': sw_metrics['omega'],
                'classification': sw_metrics['classification'],
                'C_obs': sw_metrics['C_obs'],
                'L_obs': sw_metrics['L_obs'],
                'is_significant': is_significant,
                'p_value': p_value,
                'total_trade_value': df_real[df_real['Interval'] == interval]['TradeValue'].sum()
            }
            results.append(result)
    return pd.DataFrame(results)

def create_enhanced_results_table(results_df):
    if results_df.empty:
        print("No results to display")
        return
    header = ["Interval", "Countries", "Edges", "Omega", "Classification", "C_obs", "L_obs", "Significant", "p-value", "Total Trade"]
    print(f"{'|'.join(f'{h:>12}' for h in header)}")
    print("-" * 120)
    for _, row in results_df.iterrows():
        print(f"{int(row['interval']):>12}|{int(row['n_countries']):>12}|{int(row['n_edges']):>12}|{row['omega']:.3f}|{row['classification']:>12}|{row['C_obs']:.3f}|{row['L_obs']:.3f}|{str(row['is_significant']):>12}|{row['p_value']:.3f}|{int(row['total_trade_value']):>12}")

def export_results_to_csv(results_df, filename="small_world_analysis_results.csv"):
    if not results_df.empty:
        export_df = results_df.copy()
        numeric_cols = ['omega', 'C_obs', 'L_obs', 'p_value']
        for col in numeric_cols:
            if col in export_df.columns:
                export_df[col] = export_df[col].round(3)
        export_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
        return export_df
    else:
        print("No results to export")
        return None

# Example usage:
# results_df = comprehensive_small_world_analysis(df_real)
# create_enhanced_results_table(results_df)
# export_results_to_csv(results_df)

# %%
def analyze_trade_network(df_real):
    intervals = [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
    
    for interval in intervals:
        # 1. Filter data for current interval
        interval_data = df_real[df_real['Interval'] == interval]
        datasets_by_interval.append(interval_data.copy())  # Store raw dataset
        
        # 2. Create trade network
        G = nx.Graph()
        for _, row in interval_data.iterrows():
            if row['ReporterName'] != row['PartnerName']:
                G.add_edge(row['ReporterName'], row['PartnerName'], weight=row['TradeValue'])
        
        # 3. Calculate metrics
        if G.number_of_nodes() == 0:
            continue
            
        # Basic network properties
        total_trade = interval_data['TradeValue'].sum()
        trade_value_per_interval.append((interval, total_trade))
        
        # Handle disconnected graphs
        if nx.is_connected(G):
            radius = nx.radius(G)
            diameter = nx.diameter(G)
            avg_path = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            radius = nx.radius(subgraph)
            diameter = nx.diameter(subgraph)
            avg_path = nx.average_shortest_path_length(subgraph)
            
        radius_per_interval.append((interval, radius))
        diameter_per_interval.append((interval, diameter))
        avg_path_length_per_interval.append((interval, avg_path))
        
        # Small-world metrics
        avg_clustering = nx.average_clustering(G)
        avg_clustering_per_interval.append((interval, avg_clustering))

# %%
# After running analysis
print("Average Clustering Coefficients:")
print(sorted(avg_clustering_per_interval, key=lambda x: x[0]))

print("\nNetwork Diameter Evolution:")
print(sorted(diameter_per_interval, key=lambda x: x[0]))

# %%
metrics = {}
for interval, interval_df in datasets_by_interval:
    metrics[interval] = {
        'Total_Trade': next((val for i, val in trade_value_per_interval if i == interval), None),
        'Radius': next((val for i, val in radius_per_interval if i == interval), None),
        'Diameter': next((val for i, val in diameter_per_interval if i == interval), None),
        'Avg_Path_Length': next((val for i, val in avg_path_length_per_interval if i == interval), None),
        'Avg_Clustering': next((val for i, val in avg_clustering_per_interval if i == interval), None)
    }
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

# %%
#Top 10 countries by trade value
def analyze_and_plot_top_10_countries(df):
        trade_value_by_country = df.groupby("ReporterName")["TradeValue"].sum()
        top_10_countries = trade_value_by_country.sort_values(ascending=False).head(10)

        print("Top 10 countries by TradeValue in:")
        print(top_10_countries)

        top_10_countries_df = top_10_countries.reset_index()
        top_10_countries_df.columns = ["Country", "TotalTradeValue"]

        colors = plt.cm.viridis(np.linspace(0, 1, len(top_10_countries)))

        plt.figure(figsize=(11, 6))
        plt.bar(top_10_countries.index, top_10_countries.values, color=colors)
        plt.xlabel("Country")
        plt.ylabel("Total Trade Value (in 1000 USD)")
        plt.title(f"Top 10 Countries by Trade Value in {interval} - {interval + 3}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# %%
# Loop through the list and apply the function to each dataset
for interval, interval_df in datasets_by_interval:
    analyze_and_plot_top_10_countries(interval_df)
    process_and_visualize_graph(interval, interval_df)
    
# Step 2: Plot the Number of Edges for Each Interval
intervals, edges = zip(*edges_per_interval)

plt.figure(figsize=(10, 6))
plt.plot(intervals, edges, marker='o', linestyle='-', color='b')
plt.xlabel('Interval')
plt.ylabel('Number of Edges')
plt.title('Number of Edges Formed in Each Interval')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Remove duplicates before plotting
import pandas as pd

df = pd.DataFrame(trade_value_per_interval, columns=["Interval", "Total_Trade"])
df = df.groupby("Interval", as_index=False).sum()  # or .mean() if appropriate

plt.plot(df["Interval"], df["Total_Trade"], marker='o', label="Total Trade Value")
plt.legend()
plt.title("Total Trade Value per Interval")
plt.xlabel("Interval")
plt.ylabel("Total Trade Value")
plt.show()

# %%
# Example: Visualize all interval-based metrics if they have data

if datasets_by_interval and trade_value_per_interval:
    intervals, trade_values = zip(*trade_value_per_interval)
    plt.figure(figsize=(10, 5))
    plt.plot(intervals, trade_values, marker='o', label='Total Trade Value')
    plt.xlabel('Interval')
    plt.ylabel('Total Trade Value')
    plt.title('Total Trade Value per Interval')
    plt.legend()
    plt.show()

if radius_per_interval:
    intervals, radii = zip(*radius_per_interval)
    plt.figure(figsize=(10, 5))
    plt.plot(intervals, radii, marker='o', label='Radius')
    plt.xlabel('Interval')
    plt.ylabel('Radius')
    plt.title('Radius per Interval')
    plt.legend()
    plt.show()

if diameter_per_interval:
    intervals, diameters = zip(*diameter_per_interval)
    plt.figure(figsize=(10, 5))
    plt.plot(intervals, diameters, marker='o', label='Diameter')
    plt.xlabel('Interval')
    plt.ylabel('Diameter')
    plt.title('Diameter per Interval')
    plt.legend()
    plt.show()

if avg_path_length_per_interval:
    intervals, avg_path_lengths = zip(*avg_path_length_per_interval)
    plt.figure(figsize=(10, 5))
    plt.plot(intervals, avg_path_lengths, marker='o', label='Average Path Length')
    plt.xlabel('Interval')
    plt.ylabel('Average Path Length')
    plt.title('Average Path Length per Interval')
    plt.legend()
    plt.show()

if avg_clustering_per_interval:
    intervals, avg_clusterings = zip(*avg_clustering_per_interval)
    plt.figure(figsize=(10, 5))
    plt.plot(intervals, avg_clusterings, marker='o', label='Average Clustering Coefficient')
    plt.xlabel('Interval')
    plt.ylabel('Average Clustering Coefficient')
    plt.title('Average Clustering Coefficient per Interval')
    plt.legend()
    plt.show()

# %%
def get_top_connected_countries(country_name, interval=None, top_n=10):
    """
    Returns the top connected countries (by trade value) for a given country and interval.
    If interval is None, uses the latest interval available.
    """
    
    # Remove rows where 'World' or 'European Union' is either Reporter or Partner from the entire dataset
    for idx, (i, df_) in enumerate(datasets_by_interval):
        filtered_df = df_[(df_['PartnerName'] != 'World') & (df_['PartnerName'] != 'European Union') &
                          (df_['ReporterName'] != 'World') & (df_['ReporterName'] != 'European Union')]
        datasets_by_interval[idx] = (i, filtered_df)
    # Select interval
    if interval is None:
        interval = max([i for i, _ in datasets_by_interval])
    # Find the dataframe for the given interval
    interval_df1 = None
    for i, df in datasets_by_interval:
        if i == interval:
            interval_df1 = df
            break
    if interval_df1 is None:
        print(f"No data found for interval {interval}")
        return
    
    # Remove rows where 'World' is either Reporter or Partner
    interval_df = interval_df1[(interval_df1['PartnerName'] != 'World')]

    # Filter rows where the country is either Reporter or Partner
    mask = (interval_df['ReporterName'] == country_name) | (interval_df['PartnerName'] == country_name)
    country_trades = interval_df[mask]

    # Aggregate trade values with each partner
    partners = {}
    for _, row in country_trades.iterrows():
        if row['ReporterName'] == country_name:
            partner = row['PartnerName']
        else:
            partner = row['ReporterName']
        partners[partner] = partners.get(partner, 0) + row['TradeValue']

    # Sort partners by trade value
    top_partners = sorted(partners.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_partners       
    # Remove 'World' from the top partners list if present
    top_partners_no_world = [(partner, val) for partner, val in top_partners if partner.strip() != 'World']
    print("\nTop connected countries for {country_name} in 2004-2007")
    for partner, value in top_partners_no_world:
        print(f"{partner}: {value:,.2f}")

    if top_partners_no_world:
        partner_names, trade_values = zip(*top_partners_no_world)
        plt.figure(figsize=(10, 5))
        plt.bar(partner_names, trade_values, color=plt.cm.viridis(np.linspace(0, 1, len(top_partners_no_world))))
        plt.xlabel("Country")
        plt.ylabel("Total Trade Value")
        plt.title(f"Top Connected Countries for {country_name} (2004-2007)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# %%
get_top_connected_countries('United States',interval=2004, top_n=21)

# %%
get_top_connected_countries('India',interval= 2004, top_n=21)

# %%
# Step 3: Plot the Total Trade Value for Each Interval
intervals, trade_values = zip(*trade_value_per_interval)

plt.figure(figsize=(10, 6))
plt.plot(intervals, trade_values, marker='o', linestyle='-', color='g')
plt.xlabel('Interval')
plt.ylabel('Total Trade Value (in 1000 USD)')
plt.title('Total Trade Value in Each Interval')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Step 5: Plot the Average Clustering Coefficient for Each Interval
intervals, avg_clustering = zip(*avg_clustering_per_interval)

plt.figure(figsize=(10, 6))
plt.plot(intervals, avg_clustering, marker='o', linestyle='-', color='m')
plt.xlabel('Interval')
plt.ylabel('Average Clustering Coefficient')
plt.title('Average Clustering Coefficient in Each Interval')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Step 6: Plot the Average Path Length for Each Interval
intervals, avg_path_lengths = zip(*avg_path_length_per_interval)

plt.figure(figsize=(10, 6))
plt.plot(intervals, avg_path_lengths, marker='o', linestyle='-', color='c')
plt.xlabel('Interval')
plt.ylabel('Average Path Length')
plt.title('Average Path Length in Each Interval')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Step 7: Plot the Average Path Length with the 6-Degree Rule Threshold
plt.figure(figsize=(10, 6))
plt.plot(intervals, avg_path_lengths, marker='o', linestyle='-', color='c', label='Average Path Length')
plt.axhline(y=6, color='r', linestyle='--', label='6-Degree Rule Threshold')
plt.xlabel('Interval')
plt.ylabel('Average Path Length')
plt.title('Average Path Length in Each Interval with 6-Degree Rule Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Step 8: Plot the Diameter for Each Interval
intervals, diameters = zip(*diameter_per_interval)

plt.figure(figsize=(10, 6))
plt.plot(intervals, diameters, marker='o', linestyle='-', color='y')
plt.xlabel('Interval')
plt.ylabel('Diameter')
plt.title('Diameter of the Network in Each Interval')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Calculate and plot the average degree for each interval
average_degrees = []
for (interval, interval_df), (_, num_edges) in zip(datasets_by_interval, edges_per_interval):
    num_nodes = len(set(interval_df['ReporterName']).union(set(interval_df['PartnerName'])))
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    average_degrees.append((interval, avg_degree))

intervals, avg_degrees = zip(*average_degrees)

plt.figure(figsize=(10, 6))
plt.plot(intervals, avg_degrees, marker='o', linestyle='-', color='b')
plt.xlabel('Interval')
plt.ylabel('Average Degree')
plt.title('Average Degree in Each Interval')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Print the intervals and a sample from each dataset
for interval, interval_df in datasets_by_interval:
    print(f"Interval: {interval}")
    print(interval_df.head())
    print("\n")

# %%
get_top_connected_countries("India")

# %%
get_top_connected_countries("United States")

# %%
# Python
import networkx as nx
import numpy as np

def omega_coefficient(G, n_iter=10):
    # Number of nodes and edges
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Clustering and path length for original graph
    C = nx.average_clustering(G)
    try:
        L = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        return np.nan  # Not connected
    
    # Generate random graph with same degree sequence
    degree_seq = [d for n, d in G.degree()]
    C_rand_list, L_rand_list = [], []
    for _ in range(n_iter):
        G_rand = nx.configuration_model(degree_seq)
        G_rand = nx.Graph(G_rand)  # Remove parallel edges
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
        C_rand_list.append(nx.average_clustering(G_rand))
        try:
            L_rand_list.append(nx.average_shortest_path_length(G_rand))
        except nx.NetworkXError:
            continue  # Skip if not connected
    C_rand = np.mean(C_rand_list)
    L_rand = np.mean(L_rand_list)
    
    # Generate lattice graph (ring lattice)
    k = int(round(2 * m / n))  # Average degree
    G_lattice = nx.watts_strogatz_graph(n, k, 0)
    C_latt = nx.average_clustering(G_lattice)
    L_latt = nx.average_shortest_path_length(G_lattice)
    
    # Omega coefficient
    omega = (L_rand / L - C / C_latt) / (L_rand / L_latt - C_rand / C_latt)
    print(omega)

# Example usage:
# G = nx.your_graph_here()
# omega = omega_coefficient(G)
# print("Omega coefficient:", omega)

# %%
print("Trade Value per Interval:")
for interval, value in trade_value_per_interval:
    print(f"Interval {interval} - {interval + 3}: {value:.2f} USD")

# %%
# Build the full trade network from all data
G_full = nx.Graph()
for _, row in df_real.iterrows():
    if row['ReporterName'] != row['PartnerName']:
        G_full.add_edge(row['ReporterName'], row['PartnerName'], weight=row['TradeValue'])

# Calculate metrics for the real network
num_nodes = G_full.number_of_nodes()
num_edges = G_full.number_of_edges()
avg_degree = 2 * num_edges / num_nodes
clustering_real = nx.average_clustering(G_full)

# Handle disconnected graphs for path length
if nx.is_connected(G_full):
    path_length_real = nx.average_shortest_path_length(G_full)
else:
    largest_cc = max(nx.connected_components(G_full), key=len)
    G_lcc = G_full.subgraph(largest_cc)
    path_length_real = nx.average_shortest_path_length(G_lcc)

# Generate a random graph with same number of nodes and average degree
p = avg_degree / (num_nodes - 1)
G_rand = nx.erdos_renyi_graph(num_nodes, p)
clustering_rand = nx.average_clustering(G_rand)
if nx.is_connected(G_rand):
    path_length_rand = nx.average_shortest_path_length(G_rand)
else:
    largest_cc_rand = max(nx.connected_components(G_rand), key=len)
    G_rand_lcc = G_rand.subgraph(largest_cc_rand)
    path_length_rand = nx.average_shortest_path_length(G_rand_lcc)

print(f"Full Network Nodes: {num_nodes}, Edges: {num_edges}")
print(f"Average Degree: {avg_degree:.2f}")
print(f"Clustering Coefficient (Real): {clustering_real:.4f}")
print(f"Average Path Length (Real): {path_length_real:.4f}")
print(f"Clustering Coefficient (Random): {clustering_rand:.4f}")
print(f"Average Path Length (Random): {path_length_rand:.4f}")

if clustering_real > clustering_rand and path_length_real < path_length_rand:
    print("The full trade network exhibits small-world properties.")
else:
    print("The full trade network does not exhibit strong small-world properties.")

# %%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

for _, row in df.iterrows():
    if G.has_edge(row['ReporterName'], row['PartnerName']):
        G[row['ReporterName']][row['PartnerName']]['TradeValue'] += row['TradeValue']
    else:
        G.add_edge(row['ReporterName'], row['PartnerName'], TradeValue=row['TradeValue'])

node_sizes = []
for node in G.nodes():
    total_weight = sum(G[node][neighbor]['TradeValue'] for neighbor in G.neighbors(node))
    node_sizes.append(total_weight)

max_node_size = max(node_sizes)
node_sizes = [size / max_node_size * 10 for size in node_sizes]

pos = nx.spiral_layout(G) 
node_sizes = [9 * G.degree(node) for node in G.nodes]

plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.4)
nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="black")
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_nodes(G, pos, nodelist=[max_connect_count], node_color="red", label=f"Highest: {max_connect_count}")
nx.draw_networkx_nodes(G, pos, nodelist=[min_connect_count], node_color="green", label=f"Highest: {min_connect_count}")

labels = {node: node for node in G.nodes if G.degree(node) > 20}  
nx.draw_networkx_labels(G, pos, labels, font_size=9)

plt.title("Trade Network Visualization with Node Sizes Based on Weights", fontsize=10)
plt.show()


# %%
# Visualize the full trade network (G_full) using spiral layout
plt.figure(figsize=(14, 14))
pos = nx.spiral_layout(G_full)

# Compute node sizes proportional to total TradeValue (sum of all edges for each node)
node_trade_values = {}
for u, v, d in G_full.edges(data=True):
    node_trade_values[u] = node_trade_values.get(u, 0) + d.get('weight', 0)
    node_trade_values[v] = node_trade_values.get(v, 0) + d.get('weight', 0)

# Ensure every node has a value (even if 0)
for n in G_full.nodes():
    if n not in node_trade_values:
        node_trade_values[n] = 0

max_trade = max(node_trade_values.values()) if node_trade_values else 1
min_size = 10
max_size = 400

# Make node sizes directly proportional to trade value (no max size cap)
node_sizes = [
    node_trade_values[n] / 1e9 + 10  # simple scaling for visibility
    for n in G_full.nodes()
]
node_sizes = [
    min_size + (max_size - min_size) * (node_trade_values[n] / max_trade) if max_trade > 0 else min_size
    for n in G_full.nodes()
]

nx.draw_networkx_nodes(G_full, pos, node_size=node_sizes, node_color="skyblue", alpha=0.5)
nx.draw_networkx_edges(G_full, pos, alpha=0.2, edge_color="gray")
nx.draw_networkx_labels(G_full, pos, font_size=7)
plt.title("Full Trade Network Visualization (Spiral Layout)", fontsize=16)
plt.axis('off')
plt.show()

# %%
# Find and print the top 5 hub countries (by degree) for each interval in the trade network

for interval, interval_df in datasets_by_interval:
    G = nx.Graph()
    for _, row in interval_df.iterrows():
        if row['ReporterName'] != row['PartnerName']:
            G.add_edge(row['ReporterName'], row['PartnerName'])
    degree_dict = dict(G.degree())
    top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Interval {interval}-{interval + 3}:")
    for country, degree in top_hubs:
        print(f"  {country}: {degree} connections")
    print()
    
    
    # Collect top 5 hubs for each interval
    intervals = []
    hub_countries = []
    hub_degrees = []

    for interval, interval_df in datasets_by_interval:
        G = nx.Graph()
        for _, row in interval_df.iterrows():
            if row['ReporterName'] != row['PartnerName']:
                G.add_edge(row['ReporterName'], row['PartnerName'])
        degree_dict = dict(G.degree())
        top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        for country, degree in top_hubs:
            intervals.append(interval)
            hub_countries.append(country)
            hub_degrees.append(degree)

    # Create a DataFrame for easier plotting
    hubs_df = pd.DataFrame({
        'Interval': intervals,
        'Country': hub_countries,
        'Degree': hub_degrees
    })

    plt.figure(figsize=(12, 7))
    for country in hubs_df['Country'].unique():
        country_data = hubs_df[hubs_df['Country'] == country]
        plt.plot(country_data['Interval'], country_data['Degree'], marker='o', label=country)

    plt.xlabel('Interval')
    plt.ylabel('Degree (Connections)')
    plt.title('Top 5 Hub Countries by Degree Across Intervals')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %%
def get_max_connect_count(graph):
    degree_dict = dict(graph.degree())
    max_connect_count = max(degree_dict, key=degree_dict.get)
    return max_connect_count

def get_min_connect_count(graph):
    degree_dict = dict(graph.degree())
    min_connect_count = min(degree_dict, key=degree_dict.get)
    return min_connect_count

# Usage
max_connect_count = get_max_connect_count(G)
min_connect_count = get_min_connect_count(G)

# %%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("D:/SWN Projects/trade_1988_2021.csv")

G = nx.Graph()

for _, row in df.iterrows():
    if G.has_edge(row['ReporterName'], row['PartnerName']):
        G[row['ReporterName']][row['PartnerName']]['TradeValue'] += row['TradeValue']
    else:
        G.add_edge(row['ReporterName'], row['PartnerName'], TradeValue=row['TradeValue'])

node_sizes = []
for node in G.nodes():
    total_weight = sum(G[node][neighbor]['TradeValue'] for neighbor in G.neighbors(node))
    node_sizes.append(total_weight)

max_node_size = max(node_sizes)
node_sizes = [size / max_node_size * 10 for size in node_sizes]

pos = nx.spiral_layout(G) 
node_sizes = [9 * G.degree(node) for node in G.nodes]

plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.4)
nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="black")
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_nodes(G, pos, nodelist=[max_connect_count], node_color="red", label=f"Highest: {max_connect_count}")
nx.draw_networkx_nodes(G, pos, nodelist=[min_connect_count], node_color="green", label=f"Highest: {min_connect_count}")

labels = {node: node for node in G.nodes if G.degree(node) > 20}  
nx.draw_networkx_labels(G, pos, labels, font_size=9)

plt.title("Trade Network Visualization with Node Sizes Based on Weights", fontsize=10)
plt.show()



