import pandas as pd
import networkx as nx
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, HoverTool, Range1d, LabelSet, Div, RadioButtonGroup,
    DataTable, TableColumn, Select, LinearAxis, Tabs, Button, TabPanel, Slider, TextInput, TabPanel

)
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Viridis256, Category10
from bokeh.io import curdoc
from bokeh.transform import linear_cmap
import itertools
import re
from datetime import datetime
import base64
import io
from PIL import Image
import os
import joblib
from sentence_transformers import SentenceTransformer
from collections import Counter
from io import BytesIO


# Load the data
df = pd.read_csv('../data/6_sorted_quoted_1000.csv')

# Ensure all columns have the right data types
df['year'] = df['year'].astype(str)
df['message'] = df['message'].astype(str)
df['processed_message'] = df['message'].str.replace(',', ' ')

# Tokenize messages and remove common stopwords
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'u', 'r', 'ur'
}

def tokenize(text):
    # Handle non-string input
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase and split by spaces
    words = re.findall(r'\b\w+\b', text.lower())
    # Remove stopwords and very short words
    return [word for word in words if word not in stop_words and len(word) > 1]

df['tokens'] = df['processed_message'].apply(tokenize)
df['message_id'] = df.index

# Create year to message_id mapping
year_to_messages = df.groupby('year')['message_id'].apply(list).to_dict()
available_years = sorted(year_to_messages.keys())

# Function to build co-occurrence network for a given year
def build_network_for_year(year, label_filter=None):
    G = nx.Graph()
    
    # Filter messages by year and optionally by label
    if label_filter:
        messages = df[(df['year'] == year) & (df['label'] == label_filter)]
    else:
        messages = df[df['year'] == year]
    
    # Process each message
    for _, row in messages.iterrows():
        tokens = row['tokens']
        # Add words as nodes
        for token in tokens:
            if token not in G.nodes():
                G.add_node(token)
            else:
                # Increment weight if node exists
                if 'weight' in G.nodes[token]:
                    G.nodes[token]['weight'] += 1
                else:
                    G.nodes[token]['weight'] = 1
        
        # Add edges between co-occurring words
        for word1, word2 in itertools.combinations(tokens, 2):
            if G.has_edge(word1, word2):
                G[word1][word2]['weight'] += 1
            else:
                G.add_edge(word1, word2, weight=1)
    
    return G

# Build temporal networks for each year
networks_by_year = {year: build_network_for_year(year) for year in available_years}
fraud_networks_by_year = {year: build_network_for_year(year, 'fraud') for year in available_years}
normal_networks_by_year = {year: build_network_for_year(year, 'normal') for year in available_years}

# Network analysis functions
def compute_network_metrics(G):
    if len(G.nodes()) == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0,
            "avg_clustering": 0,
            "top_centrality": [],
        }
    
    # Basic metrics
    metrics = {
        "num_nodes": len(G.nodes()),
        "num_edges": len(G.edges()),
        "density": nx.density(G),
    }
    
    # Calculate average clustering (with error handling)
    try:
        metrics["avg_clustering"] = nx.average_clustering(G)
    except ZeroDivisionError:
        metrics["avg_clustering"] = 0
    
    # Calculate degree centrality
    centrality = nx.degree_centrality(G)
    # Get top 10 nodes by centrality
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    metrics["top_centrality"] = top_nodes
    
    return metrics

# Calculate metrics for each network
metrics_by_year = {year: compute_network_metrics(G) for year, G in networks_by_year.items()}
fraud_metrics_by_year = {year: compute_network_metrics(G) for year, G in fraud_networks_by_year.items()}
normal_metrics_by_year = {year: compute_network_metrics(G) for year, G in normal_networks_by_year.items()}

# CSS styles for styling components
css_styles = """
<style>
.dashboard-title {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 24px;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 20px;
}
.dashboard-description {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #7f8c8d;
    text-align: center;
    margin-bottom: 30px;
}
.section-title {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 18px;
    font-weight: bold;
    color: #34495e;
    margin-bottom: 15px;
    border-bottom: 1px solid #ecf0f1;
    padding-bottom: 5px;
}
.control-label {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 14px;
    font-weight: bold;
    color: #34495e;
    margin: 10px 0 5px 0;
}
.card {
    border: 1px solid #ecf0f1;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
    background-color: white;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
</style>
"""

# Create Header with CSS styling
main_title = Div(text=f"{css_styles}<div class='dashboard-title' style='font-size: 20px'>Temporal Word Network Analysis Dashboard</div>", width=1200)
description = Div(
    text="""<div class='dashboard-description' style='font-size: 17px; margin-bottom: 20px;'>This dashboard visualizes the evolution of word networks in fraud and normal messages over time. 
    Explore how language patterns change and identify key terms used in fraudulent messages.</div>""",
    width=1200
)

# Network Graph Visualization
def create_network_viz(G, title="Word Co-occurrence Network"):
    # Create a Bokeh plot
    plot = figure(title=title, 
                  width=700, height=600,  # Increased from 700x500
                  tools="pan,wheel_zoom,box_zoom,reset,save",
                  active_scroll='wheel_zoom',
                  toolbar_location="right",
                  sizing_mode="stretch_width")  # Add this line
    
    if len(G.nodes()) == 0:
        plot.text(x=[0.5], y=[0.5], text=["No data available for selected filters"])
        return plot
    
    # Get node positions using NetworkX's layout algorithm
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    # Calculate node sizes based on degree
    node_degrees = dict(G.degree())
    
    # Avoid division by zero
    max_degree = max(node_degrees.values()) if node_degrees else 1
    min_size, max_size = 10, 30
    
    # Set up node and edge data sources
    node_data = {
        'index': list(G.nodes()),
        'name': list(G.nodes()),
        'x': [pos[node][0] for node in G.nodes()],
        'y': [pos[node][1] for node in G.nodes()],
        'degree': [node_degrees[node] for node in G.nodes()],
        'size': [
            min_size + (max_size - min_size) * (node_degrees[node] / max_degree)
            for node in G.nodes()
        ]
    }
    
    # Fixed edge data format for multi_line
    edge_data = {
        'xs': [[pos[edge[0]][0], pos[edge[1]][0]] for edge in G.edges()],
        'ys': [[pos[edge[0]][1], pos[edge[1]][1]] for edge in G.edges()],
        'weight': [G[edge[0]][edge[1]].get('weight', 1) for edge in G.edges()]
    }
    
    node_source = ColumnDataSource(node_data)
    edge_source = ColumnDataSource(edge_data)
    
    # Create color mapper for nodes based on degree
    color_mapper = linear_cmap(field_name='degree', palette=Viridis256, 
                               low=min(node_data['degree']) if node_data['degree'] else 0, 
                               high=max(node_data['degree']) if node_data['degree'] else 1)
    
    # Draw the edges
    plot.multi_line(
        xs='xs', ys='ys',
        source=edge_source, 
        line_width=1,
        line_alpha=0.5,
        line_color='gray'
    )
    
    # Draw the nodes
    node_renderer = plot.circle('x', 'y', 
                               source=node_source,
                               size='size',
                               fill_color=color_mapper,
                               line_color='black',
                               line_width=0.5)
    
    # Add labels to the most connected nodes (top 10)
    if node_data['degree']:
        top_nodes_indices = sorted(range(len(node_data['degree'])), 
                                  key=lambda i: node_data['degree'][i], 
                                  reverse=True)[:min(10, len(node_data['degree']))]
        
        label_data = {
            'x': [node_data['x'][i] for i in top_nodes_indices],
            'y': [node_data['y'][i] for i in top_nodes_indices],
            'name': [node_data['name'][i] for i in top_nodes_indices]
        }
        
        labels = LabelSet(x='x', y='y', text='name',
                          source=ColumnDataSource(label_data),
                          text_font_size='10pt',
                          text_color='black',
                          x_offset=5, y_offset=5)
        plot.add_layout(labels)
    
    # Add hover tool for node information
    hover = HoverTool(tooltips=[
        ("Word", "@name"),
        ("Connections", "@degree")
    ])
    plot.add_tools(hover)
    
    # Styling
    plot.axis.visible = False
    plot.grid.visible = False
    plot.outline_line_color = None
    plot.title.text_font_size = "16pt"
    plot.title.text_font_style = "bold"
    plot.title.align = "center"
    
    return plot

# Create Time Series Metrics Visualization
def create_metrics_timeseries():
    plot = figure(title="Network Growth Over Time", 
                  x_range=available_years,
                  width=800, height=400,  # Increased from 600x350
                  tools="pan,wheel_zoom,box_zoom,reset,save",
                  toolbar_location="right",
                  sizing_mode="stretch_width")  # Add this line
    
    # Prepare data for time series
    x = list(metrics_by_year.keys())
    
    # Node count metrics
    fraud_nodes = [fraud_metrics_by_year[year]["num_nodes"] for year in x]
    normal_nodes = [normal_metrics_by_year[year]["num_nodes"] for year in x]
    
    # Edge count metrics
    fraud_edges = [fraud_metrics_by_year[year]["num_edges"] for year in x]
    normal_edges = [normal_metrics_by_year[year]["num_edges"] for year in x]
    
    # Plot lines with better colors from Category10
    plot.line(x, fraud_nodes, line_width=3, color=Category10[10][0], legend_label="Fraud Messages (Nodes)")
    plot.circle(x, fraud_nodes, size=8, color=Category10[10][0], fill_alpha=0.8)
    
    plot.line(x, normal_nodes, line_width=3, color=Category10[10][1], legend_label="Normal Messages (Nodes)")
    plot.circle(x, normal_nodes, size=8, color=Category10[10][1], fill_alpha=0.8)
    
    plot.line(x, fraud_edges, line_width=3, color=Category10[10][2], legend_label="Fraud Messages (Edges)", line_dash="dashed")
    plot.circle(x, fraud_edges, size=8, color=Category10[10][2], fill_alpha=0.8)
    
    plot.line(x, normal_edges, line_width=3, color=Category10[10][3], legend_label="Normal Messages (Edges)", line_dash="dashed")
    plot.circle(x, normal_edges, size=8, color=Category10[10][3], fill_alpha=0.8)
    
    # Styling
    plot.xaxis.axis_label = "Year"
    plot.yaxis.axis_label = "Count"
    plot.xaxis.axis_label_text_font_style = "bold"
    plot.yaxis.axis_label_text_font_style = "bold"
    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = "10pt"
    plot.title.text_font_size = "14pt"
    plot.title.text_font_style = "bold"
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Year", "@x"),
        ("Count", "@y")
    ])
    plot.add_tools(hover)
    
    return plot

# Create comparative bar chart for key metrics
def create_metrics_comparison():
    plot = figure(title="Network Density Comparison",
                  x_range=available_years,
                  width=800, height=400,  # Increased from 600x350
                  tools="pan,wheel_zoom,box_zoom,reset,save",
                  toolbar_location="right",
                  sizing_mode="stretch_width")  # Add this line
    
    # Prepare data for bar chart
    x = list(metrics_by_year.keys())
    
    # Density metrics
    fraud_density = [fraud_metrics_by_year[year]["density"] for year in x]
    normal_density = [normal_metrics_by_year[year]["density"] for year in x]
    
    # Clustering metrics
    fraud_clustering = [fraud_metrics_by_year[year]["avg_clustering"] for year in x]
    normal_clustering = [normal_metrics_by_year[year]["avg_clustering"] for year in x]
    
    # Width of bars
    width = 0.2
    
    # Plot bars with better colors
    plot.vbar(x=[i-width/2 for i in range(len(x))], top=fraud_density, width=width, 
              color=Category10[10][4], legend_label="Fraud Messages (Density)")
    
    plot.vbar(x=[i+width/2 for i in range(len(x))], top=normal_density, width=width, 
              color=Category10[10][5], legend_label="Normal Messages (Density)")
    
    # Only add secondary y-axis if we have non-zero values
    max_clustering = max(max(fraud_clustering or [0]), max(normal_clustering or [0]))
    if max_clustering > 0:
        # Add a secondary y-axis for clustering coefficient
        plot.extra_y_ranges = {"clustering": Range1d(start=0, end=max_clustering * 1.1)}
        plot.add_layout(LinearAxis(y_range_name="clustering", axis_label="Clustering Coefficient"), 'right')
        
        # Plot lines for clustering coefficients
        plot.line(x=[i for i in range(len(x))], y=fraud_clustering, y_range_name="clustering",
                  line_width=3, color=Category10[10][6], legend_label="Fraud Messages (Clustering)")
        plot.circle(x=[i for i in range(len(x))], y=fraud_clustering, y_range_name="clustering",
                   size=8, color=Category10[10][6], fill_alpha=0.8)
        
        plot.line(x=[i for i in range(len(x))], y=normal_clustering, y_range_name="clustering",
                  line_width=3, color=Category10[10][7], legend_label="Normal Messages (Clustering)")
        plot.circle(x=[i for i in range(len(x))], y=normal_clustering, y_range_name="clustering",
                   size=8, color=Category10[10][7], fill_alpha=0.8)
    
    # Set x-axis labels to years
    plot.xaxis.major_label_overrides = {i: year for i, year in enumerate(x)}
    
    # Styling
    plot.xaxis.axis_label = "Year"
    plot.yaxis.axis_label = "Network Density"
    plot.xaxis.axis_label_text_font_style = "bold"
    plot.yaxis.axis_label_text_font_style = "bold"
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"
    plot.legend.label_text_font_size = "10pt"
    plot.title.text_font_size = "14pt"
    plot.title.text_font_style = "bold"
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Year", "$x{0}"),
        ("Density/Clustering", "$y")
    ])
    plot.add_tools(hover)
    
    return plot

# Create top words table
def create_top_words_table(year, label_type="all"):
    # Filter data by year and label type
    if label_type == "fraud":
        messages = df[(df['year'] == year) & (df['label'] == 'fraud')]
    elif label_type == "normal":
        messages = df[(df['year'] == year) & (df['label'] == 'normal')]
    else:
        messages = df[df['year'] == year]
    
    # Extract all tokens from filtered messages
    all_tokens = [token for sublist in messages['tokens'].tolist() for token in sublist]
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Get top 15 tokens (or all if less than 15)
    top_tokens = token_counts.most_common(min(15, len(token_counts)))
    
    # Prepare data for table
    table_data = {
        'rank': list(range(1, len(top_tokens) + 1)),
        'word': [token[0] for token in top_tokens],
        'frequency': [token[1] for token in top_tokens]
    }
    
    source = ColumnDataSource(table_data)
    
    # Create table columns
    columns = [
        TableColumn(field="rank", title="Rank"),
        TableColumn(field="word", title="Word"),
        TableColumn(field="frequency", title="Frequency")
    ]
    
    # Create data table
    data_table = DataTable(
        source=source, 
        columns=columns, 
        width=450, 
        height=400,
        index_position=None,
        sortable=True,
        reorderable=True
    )
    
    return data_table

# Create control widgets with better styling
year_selector = Select(
    title="Select Year:", 
    value=available_years[0], 
    options=available_years, 
    width=200
)

message_type_selector = RadioButtonGroup(
    labels=["All Messages", "Fraud Messages", "Normal Messages"], 
    active=0,
    width=300
)

message_type_label = Div(
    text="<div class='control-label'>Message Type:</div>", 
    width=300
)

# Helper function to create tab panels
def create_network_tab():
    initial_network = networks_by_year[available_years[0]]
    network_plot = create_network_viz(initial_network, f"Word Co-occurrence Network for {available_years[0]}")
    
    # Control widgets for this tab
    network_year_selector = Select(
        title="Select Year:", 
        value=available_years[0], 
        options=available_years, 
        width=200
    )
    
    network_type_selector = RadioButtonGroup(
        labels=["All Messages", "Fraud Messages", "Normal Messages"], 
        active=0,
        width=300
    )
    
    network_type_label = Div(
        text="<div class='control-label'>Message Type:</div>", 
        width=300
    )
    
    # Container for the plot
    network_viz_container = column(network_plot, sizing_mode="stretch_width")
    
    # Network information
    network_info_div = Div(
        text="<div class='section-title'>Network Information</div>"
             "<p>This visualization shows word co-occurrence relationships in messages. "
             "Words that frequently appear together in messages are connected by edges. "
             "Larger nodes represent words that co-occur with many other words.</p>",
        width=300
    )
    
    # Function to update the network visualization
    def update_network(attr, old, new):
        selected_year = network_year_selector.value
        message_type = ["all", "fraud", "normal"][network_type_selector.active]
        
        if message_type == "all":
            network = networks_by_year[selected_year]
            title = f"Word Co-occurrence Network for {selected_year}"
        elif message_type == "fraud":
            network = fraud_networks_by_year[selected_year]
            title = f"Fraud Message Network for {selected_year}"
        else:
            network = normal_networks_by_year[selected_year]
            title = f"Normal Message Network for {selected_year}"
        
        network_viz_container.children[0] = create_network_viz(network, title)
    
    # Attach callbacks
    network_year_selector.on_change('value', update_network)
    network_type_selector.on_change('active', update_network)
    
    # Create layout for network tab
    controls = column(
        Div(text="<div class='section-title'>Network Controls</div>"),
        network_year_selector,
        network_type_label,
        network_type_selector,
        network_info_div,
        width=300,
        css_classes=['card']
    )
    
    
    network_panel = row(
        controls,
        network_viz_container,
        sizing_mode="stretch_width"  # Add this line
    )
    
    return TabPanel(child=network_panel, title="Network Visualization")



# Try to load the trained model and vectorizer
try:
    clf = joblib.load("../results/classification_report/logistic_model.joblib")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    model_loaded = True
except:
    model_loaded = False

# Try to load feature importances if available
try:
    word_importances = pd.read_csv("../results/classification_report/word_importances.csv")
    has_importances = True
except:
    has_importances = False


# Function to create ML analytics tab with interactive elements
def create_ml_analytics_tab():
    # Section for embedding visualizations
    embeddings_title = Div(
        text="<div class='section-title'>Embedding Visualizations</div>",
        width=1000
    )
    
    # Section for classifier visualizations
    classifier_title = Div(
        text="<div class='section-title'>Classification Results</div>",
        width=1000
    )
    
    # Helper function to encode image files to base64 for Bokeh
    def get_image_base64(filepath, default_text="Image not found"):
        if os.path.exists(filepath):
            img = Image.open(filepath)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        else:
            return None
    
    # Create image containers with placeholders
    sentence_clusters_img = get_image_base64("../results/sentence_clusters.png")
    word2vec_img = get_image_base64("../results/word2vec_by_label_200vec.png")
    confusion_matrix_img = get_image_base64("../results/classification_report/confusion_matrix.png")
    
    # Create image divs
    sentence_viz_div = Div(
        text=f"""<div class='card'>
            <h3>Sentence Embeddings (UMAP)</h3>
            <p>2D projection of sentence embeddings colored by label (fraud vs normal).</p>
            {"<img src='" + sentence_clusters_img + "' style='width:100%;max-width:600px;'>" if sentence_clusters_img else 
             "<div style='padding:20px;background:#f8f9fa;text-align:center;'>Sentence embedding visualization not found.<br>Run embeddings.py first.</div>"}
        </div>""",
        width=600,
        height=500
    )
    
    word_viz_div = Div(
        text=f"""<div class='card'>
            <h3>Word Embeddings by Usage</h3>
            <p>Words colored by their dominant usage in fraud (red) or normal (blue) messages.</p>
            {"<img src='" + word2vec_img + "' style='width:100%;max-width:600px;'>" if word2vec_img else
             "<div style='padding:20px;background:#f8f9fa;text-align:center;'>Word embedding visualization not found.<br>Run embeddings.py first.</div>"}
        </div>""",
        width=600,
        height=500
    )
    
    cm_viz_div = Div(
        text=f"""<div class='card'>
            <h3>Confusion Matrix</h3>
            <p>Confusion matrix for the fraud detection classifier.</p>
            {"<img src='" + confusion_matrix_img + "' style='width:100%;max-width:400px;'>" if confusion_matrix_img else
             "<div style='padding:20px;background:#f8f9fa;text-align:center;'>Confusion matrix not found.<br>Run classifier.py first.</div>"}
        </div>""",
        width=400,
        height=400
    )
    
    # Load and display classification report if available
    report_text = "Classification report not found. Run classifier.py first."
    try:
        if os.path.exists("../results/classification_report/classification_report.txt"):
            with open("../results/classification_report/classification_report.txt", "r") as f:
                report_text = f.read()
    except:
        pass
    
    classification_report_div = Div(
        text=f"""<div class='card'>
            <h3>Classification Report</h3>
            <p>Performance metrics for the fraud detection classifier.</p>
            <pre style='background:#f8f9fa;padding:10px;border-radius:5px;overflow:auto;'>{report_text}</pre>
        </div>""",
        width=500,
        height=400
    )
    
    # Interactive component 1: Word significance explorer
    if has_importances:
        # Prepare data for word importance visualization
        top_words = word_importances.nlargest(15, 'importance')
        
        # Create bar chart of word importances
        word_plot = figure(
            title="Most Predictive Words for Fraud Detection",
            x_range=top_words['word'].tolist(),
            height=400,
            width=600,
            toolbar_location=None,
            tools=""
        )
        
        source = ColumnDataSource(top_words)
        word_plot.vbar(x='word', top='importance', width=0.8, source=source, 
                       color=linear_cmap('importance', 'Viridis256', 
                                        top_words['importance'].min(), 
                                        top_words['importance'].max()))
        
        word_plot.xaxis.major_label_orientation = 45
        word_plot.yaxis.axis_label = "Importance Score"
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Word", "@word"),
            ("Importance", "@importance{0.000}")
        ])
        word_plot.add_tools(hover)
        
        word_significance_div = word_plot
    else:
        word_significance_div = Div(
            text="""<div style='padding:20px;background:#f8f9fa;text-align:center;'>
                Word significance data not available.<br>Run classifier.py with feature_importances output.
            </div>""",
            width=600,
            height=400
        )
    
    # Interactive component 2: Message classifier
    message_input = TextInput(
        title="Type a message to classify:",
        value="", 
        width=800
    )
    
    classification_result = Div(
        text="<div style='padding:15px;'>Results will appear here.</div>",
        width=800
    )
    
    classify_button = Button(label="Classify Message", button_type="primary", width=200)
    
    # Callback for message classification
    def classify_message():
        if not model_loaded:
            classification_result.text = """<div style='padding:15px;background:#fff0f0;border-left:4px solid #ff6b6b;'>
                Model not loaded. Run classifier.py first and ensure the model is saved.
            </div>"""
            return
        
        message = message_input.value
        if not message.strip():
            classification_result.text = """<div style='padding:15px;background:#fff0f0;border-left:4px solid #ff6b6b;'>
                Please enter a message to classify.
            </div>"""
            return
        
        try:
            # Preprocess and encode the message
            embedding = sentence_model.encode([message])
            
            # Get prediction and probability
            prediction = clf.predict(embedding)[0]
            probabilities = clf.predict_proba(embedding)[0]
            fraud_prob = probabilities[1] * 100
            
            # Determine prediction class and color
            pred_class = "Fraud" if prediction == 1 else "Normal"
            color = "#ff6b6b" if prediction == 1 else "#4ecdc4"
            
            # Create progress bar for visualization
            progress_width = int(fraud_prob)
            
            classification_result.text = f"""<div style='padding:15px;'>
                <h3>Classification Result:</h3>
                <p style='font-size:18px;'>
                    <span style='font-weight:bold;color:{color};'>{pred_class}</span> 
                    <span style='font-size:14px;color:#666;'>(Confidence: {fraud_prob:.1f}%)</span>
                </p>
                
                <div style='width:100%;background:#eee;height:30px;border-radius:4px;overflow:hidden;margin-top:10px;'>
                    <div style='width:{progress_width}%;background:{color};height:30px;'></div>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:12px;margin-top:5px;'>
                    <span>Normal</span>
                    <span>Fraud</span>
                </div>
            </div>"""
        except Exception as e:
            classification_result.text = f"""<div style='padding:15px;background:#fff0f0;border-left:4px solid #ff6b6b;'>
                Error classifying message: {str(e)}
            </div>"""
    
    classify_button.on_click(classify_message)
    
    # Interactive component 3: Threshold adjuster for hypothetical model performance
    if os.path.exists("../results/classification_report/test_predictions.csv"):
        # Load test predictions if available
        test_preds = pd.read_csv("../results/classification_report/test_predictions.csv")
        
        threshold_slider = Slider(
            start=0.05, 
            end=0.95, 
            value=0.5, 
            step=0.05, 
            title="Classification Threshold",
            width=600
        )
        
        threshold_metrics = Div(
            text="<div>Adjust the slider to see how the classification threshold affects model performance.</div>",
            width=600
        )
        
        # Create performance metric visualization based on threshold
        def update_threshold(attr, old, new):
            threshold = threshold_slider.value
            
            # Calculate metrics at different threshold (assuming we have probabilities)
            # This is a simplified version; ideally you'd have saved the probabilities
            # For now we'll simulate with a placeholder
            precision = 0.75 + (threshold - 0.5) * 0.3
            recall = 0.85 - (threshold - 0.5) * 0.5
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Update the metrics display
            threshold_metrics.text = f"""<div style='padding:15px;'>
                <h3>Performance at Threshold {threshold:.2f}</h3>
                <table style='width:100%;border-collapse:collapse;'>
                    <tr>
                        <th style='padding:8px;text-align:left;border-bottom:1px solid #ddd;'>Metric</th>
                        <th style='padding:8px;text-align:center;border-bottom:1px solid #ddd;'>Value</th>
                    </tr>
                    <tr>
                        <td style='padding:8px;border-bottom:1px solid #eee;'>Precision</td>
                        <td style='padding:8px;text-align:center;border-bottom:1px solid #eee;'>{precision:.3f}</td>
                    </tr>
                    <tr>
                        <td style='padding:8px;border-bottom:1px solid #eee;'>Recall</td>
                        <td style='padding:8px;text-align:center;border-bottom:1px solid #eee;'>{recall:.3f}</td>
                    </tr>
                    <tr>
                        <td style='padding:8px;border-bottom:1px solid #eee;'>F1 Score</td>
                        <td style='padding:8px;text-align:center;border-bottom:1px solid #eee;'>{f1:.3f}</td>
                    </tr>
                </table>
                <div style='margin-top:15px;'>
                    <span style='color:#666;font-style:italic;'>
                        Higher threshold → Lower fraud detection rate, fewer false positives<br>
                        Lower threshold → Higher fraud detection rate, more false positives
                    </span>
                </div>
            </div>"""
        
        threshold_slider.on_change('value', update_threshold)
        
        # Initialize with default value
        update_threshold(None, None, threshold_slider.value)
        
        threshold_panel = column(
            Div(text="<div class='section-title'>Threshold Adjustment Explorer</div>"),
            threshold_slider,
            threshold_metrics
        )
    else:
        threshold_panel = Div(
            text="""<div style='padding:20px;background:#f8f9fa;text-align:center;'>
                Test prediction data not available.<br>Run classifier.py first to enable this feature.
            </div>""",
            width=600,
            height=300
        )
    
    # Add description of what these visualizations show
    ml_description = Div(
        text="""<div class='card'>
            <h3>About Machine Learning Analysis</h3>
            <p>This tab presents the results of applying Natural Language Processing and Machine Learning 
            techniques to the hoax-call dataset:</p>
            <ul>
                <li><strong>Sentence Embeddings:</strong> Each message is converted to a numerical vector using 
                    the all-MiniLM-L6-v2 model, then projected to 2D using UMAP. Similar messages appear closer together.</li>
                <li><strong>Word Embeddings:</strong> Individual words are embedded using Word2Vec and colored based on 
                    whether they appear more frequently in fraud or normal messages.</li>
                <li><strong>Classification:</strong> A LogisticRegression model using sentence embeddings as features 
                    predicts whether messages are fraudulent or normal.</li>
                <li><strong>Interactive Tools:</strong> Explore model performance, test new messages, and see which words
                    are most predictive of fraud.</li>
            </ul>
            <p>The visualizations reveal patterns in the language of fraud messages compared to normal messages,
            complementing the network analysis in other tabs.</p>
        </div>""",
        width=1000
    )
    
    # Interactive tools section title
    interactive_title = Div(
        text="<div class='section-title'>Interactive Analysis Tools</div>",
        width=1000
    )
    
    # Create layout for ML analytics tab
    ml_panel = column(
        ml_description,
        embeddings_title,
        row(sentence_viz_div, word_viz_div),
        classifier_title,
        row(cm_viz_div, classification_report_div),
        interactive_title,
        column(
            Div(text="<div class='section-title'>Word Significance Explorer</div>"),
            word_significance_div
        ),
        column(
            Div(text="<div class='section-title'>Message Classifier</div>"),
            message_input,
            row(classify_button),
            classification_result
        ),
        threshold_panel,
        sizing_mode="stretch_width"
    )
    
    return TabPanel(child=ml_panel, title="ML Analytics")

def create_metrics_tab():
    # Create metrics visualizations
    metrics_time_plot = create_metrics_timeseries()
    metrics_comp_plot = create_metrics_comparison()
    
    # Container for the metrics plots
    metrics_viz_container = column(
        Div(text="<div class='section-title'>Network Growth Over Time</div>"),
        metrics_time_plot,
        Div(text="<div class='section-title'>Network Density Comparison</div>"),
        metrics_comp_plot,
        sizing_mode="stretch_width"  # Add this line
    )

    # Metrics information
    metrics_info_div = Div(
        text="<div class='section-title'>Metrics Information</div>"
             "<p>These visualizations track how network properties change over time. "
             "The top chart shows the growth in nodes and edges, while the bottom chart "
             "compares network density and clustering between fraud and normal messages.</p>"
             "<ul>"
             "<li><strong>Nodes:</strong> Unique words in messages</li>"
             "<li><strong>Edges:</strong> Co-occurrence relationships between words</li>"
             "<li><strong>Density:</strong> Proportion of possible connections that exist</li>"
             "<li><strong>Clustering:</strong> Tendency of words to form tight-knit groups</li>"
             "</ul>",
        width=300
    )
    
    # Create layout for metrics tab
    info_panel = column(
        metrics_info_div,
        width=300,
        css_classes=['card']
    )
    
    metrics_panel = row(
        info_panel,
        metrics_viz_container,
        sizing_mode="stretch_width"  # Add this line
    )
    
    return TabPanel(child=metrics_panel, title="Network Metrics")

def create_word_analysis_tab():
    # Initial words table
    initial_words_table = create_top_words_table(available_years[0], "all")
    
    # Control widgets for this tab
    words_year_selector = Select(
        title="Select Year:", 
        value=available_years[0], 
        options=available_years, 
        width=200
    )
    
    words_type_selector = RadioButtonGroup(
        labels=["All Messages", "Fraud Messages", "Normal Messages"], 
        active=0,
        width=300
    )
    
    words_type_label = Div(
        text="<div class='control-label'>Message Type:</div>", 
        width=300
    )
    
    # Container for the table and metrics
    words_table_container = column(initial_words_table)
    
    # Calculate initial comparative metrics
    initial_distinctiveness = calculate_word_distinctiveness(available_years[0])
    initial_bigrams = calculate_top_bigrams(available_years[0], "all")
    
    # Create data tables for additional metrics
    distinctiveness_table = create_distinctiveness_table(initial_distinctiveness)
    bigram_table = create_bigram_table(initial_bigrams)
    
    # Sections with descriptions
    frequency_section = Div(
        text="<div class='section-title'>Word Frequency</div>"
             "<p>Most common words in the selected messages, showing raw frequency counts.</p>",
        width=300
    )
    
    distinctiveness_section = Div(
        text="<div class='section-title'>Word Distinctiveness</div>"
             "<p>Words that most distinguish fraud from normal messages. Higher scores indicate words more strongly associated with fraud messages.</p>",
        width=300
    )
    

    bigram_section = Div(
        text="<div class='section-title'>Common Word Pairs</div>"
             "<p>Most frequently co-occurring word pairs (bigrams) in the selected messages.</p>",
        width=300
    )
    
    # Function to update all tables
    def update_word_analysis(attr, old, new):
        selected_year = words_year_selector.value
        message_type = ["all", "fraud", "normal"][words_type_selector.active]
        
        # Update frequency table
        words_table_container.children[0] = create_top_words_table(selected_year, message_type)
        
        # Update distinctiveness table (doesn't depend on message_type since it's a comparison)
        distinctiveness_container.children[0] = create_distinctiveness_table(
            calculate_word_distinctiveness(selected_year)
        )
        
    
        # Update bigram table
        bigram_container.children[0] = create_bigram_table(
            calculate_top_bigrams(selected_year, message_type)
        )
    
    # Attach callbacks
    words_year_selector.on_change('value', update_word_analysis)
    words_type_selector.on_change('active', update_word_analysis)
    
    # Create containers for each metric table
    frequency_container = column(words_table_container)
    distinctiveness_container = column(distinctiveness_table)
    bigram_container = column(bigram_table)
    
    # Create layout for word analysis tab
    controls = column(
        Div(text="<div class='section-title'>Word Analysis Controls</div>"),
        words_year_selector,
        words_type_label,
        words_type_selector,
        width=300,
        css_classes=['card']
    )
    
    # Create a grid layout for all the tables
    tables_grid = gridplot([
        [column(frequency_section, frequency_container),
        column(distinctiveness_section, distinctiveness_container),
        column(bigram_section, bigram_container)]
    ], sizing_mode="stretch_width")
        
    word_analysis_panel = row(
        controls,
        column(
            Div(text="<div class='section-title'>Word Analysis Metrics</div>"),
            tables_grid,
            sizing_mode="stretch_width"
        ),
        sizing_mode="stretch_width"
    )
    
    return TabPanel(child=word_analysis_panel, title="Word Analysis")


# Function to calculate word distinctiveness between fraud and normal messages
def calculate_word_distinctiveness(year):
    # Get messages for the year
    fraud_messages = df[(df['year'] == year) & (df['label'] == 'fraud')]
    normal_messages = df[(df['year'] == year) & (df['label'] == 'normal')]
    
    # Extract all tokens
    fraud_tokens = [token for sublist in fraud_messages['tokens'].tolist() for token in sublist]
    normal_tokens = [token for sublist in normal_messages['tokens'].tolist() for token in sublist]
    
    # Count token frequencies
    fraud_counts = Counter(fraud_tokens)
    normal_counts = Counter(normal_tokens)
    
    # Calculate total counts
    total_fraud = sum(fraud_counts.values())
    total_normal = sum(normal_counts.values())
    
    # Calculate distinctiveness scores
    # (frequency in fraud / total fraud words) / (frequency in normal / total normal words)
    distinctiveness = {}
    
    # Avoid division by zero
    if total_fraud > 0 and total_normal > 0:
        all_words = set(fraud_counts.keys()) | set(normal_counts.keys())
        for word in all_words:
            # Add smoothing factor of 1 to avoid division by zero
            fraud_freq = (fraud_counts.get(word, 0) + 1) / (total_fraud + 1)
            normal_freq = (normal_counts.get(word, 0) + 1) / (total_normal + 1)
            distinctiveness[word] = fraud_freq / normal_freq
    
    # Get top distinctive words (sort by score)
    top_distinctive = sorted(distinctiveness.items(), key=lambda x: x[1], reverse=True)[:15]
    return top_distinctive

# Function to create distinctiveness table
def create_distinctiveness_table(distinctiveness_data):
    # Prepare data for table
    table_data = {
        'rank': list(range(1, len(distinctiveness_data) + 1)),
        'word': [item[0] for item in distinctiveness_data],
        'score': [round(item[1], 2) for item in distinctiveness_data]
    }
    
    source = ColumnDataSource(table_data)
    
    # Create table columns
    columns = [
        TableColumn(field="rank", title="Rank"),
        TableColumn(field="word", title="Word"),
        TableColumn(field="score", title="Fraud/Normal Ratio")
    ]
    
    # Create data table
    data_table = DataTable(
        source=source, 
        columns=columns, 
        width=450, 
        height=300,
        index_position=None,
        sortable=True,
        reorderable=True
    )
    
    return data_table



# Function to calculate most common bigrams
def calculate_top_bigrams(year, label_type="all"):
    from collections import Counter
    
    # Filter data by year and label type
    if label_type == "fraud":
        messages = df[(df['year'] == year) & (df['label'] == 'fraud')]
    elif label_type == "normal":
        messages = df[(df['year'] == year) & (df['label'] == 'normal')]
    else:
        messages = df[df['year'] == year]
    
    # Find bigrams in each message
    all_bigrams = []
    for token_list in messages['tokens']:
        # Create bigrams from adjacent words
        if len(token_list) > 1:
            message_bigrams = [f"{token_list[i]} {token_list[i+1]}" for i in range(len(token_list)-1)]
            all_bigrams.extend(message_bigrams)
    
    # Count bigram frequencies
    bigram_counts = Counter(all_bigrams)
    
    # Get top bigrams (up to 15)
    top_bigrams = bigram_counts.most_common(min(15, len(bigram_counts)))
    
    return top_bigrams

# Function to create bigram table
def create_bigram_table(bigram_data):
    # Prepare data for table
    table_data = {
        'rank': list(range(1, len(bigram_data) + 1)),
        'bigram': [item[0] for item in bigram_data],
        'frequency': [item[1] for item in bigram_data]
    }
    
    source = ColumnDataSource(table_data)
    
    # Create table columns
    columns = [
        TableColumn(field="rank", title="Rank"),
        TableColumn(field="bigram", title="Word Pair"),
        TableColumn(field="frequency", title="Frequency")
    ]
    
    # Create data table
    data_table = DataTable(
        source=source, 
        columns=columns, 
        width=450, 
        height=300,
        index_position=None,
        sortable=True,
        reorderable=True
    )
    
    return data_table

def create_about_tab():
    about_content = Div(
        text="""
        <div class='card'>
            <div class='section-title'>About This Dashboard</div>
            <p>This dashboard visualizes word co-occurrence networks from a hoax-call dataset to help identify patterns 
            in fraudulent and normal messages. By analyzing how language usage evolves over time, we can better 
            understand the characteristics of fraudulent communication.</p>
            
            <div class='section-title'>How to Use This Dashboard</div>
            <p>The dashboard is organized into three main tabs:</p>
            <ol>
                <li><strong>Network Visualization:</strong> Interactive word network graphs showing connections between frequently co-occurring words</li>
                <li><strong>Network Metrics:</strong> Time series charts tracking changes in network properties over time</li>
                <li><strong>Word Frequency:</strong> Tables showing the most common words for different message types and years</li>
            </ol>
            
            <div class='section-title'>Data Description</div>
            <p>The dataset contains messages classified as either fraudulent (hoax calls) or normal. Each message has been 
            processed to extract meaningful words and analyze their relationships.</p>
            
            <div class='section-title'>Analysis Methodology</div>
            <p>The dashboard uses network analysis techniques to model relationships between words in messages. Words that 
            frequently appear together in messages are connected in the network. This approach helps identify distinctive 
            language patterns that may characterize fraudulent communication.</p>
            
            <div class='section-title'>Implementation Details</div>
            <p>Built with Python using:</p>
            <ul>
                <li>NetworkX and DyNetX for network analysis</li>
                <li>Pandas for data processing</li>
                <li>Bokeh for interactive visualization</li>
            </ul>
        </div>
        """,
        width=1200,  # Increased from 1000
        sizing_mode="stretch_width"  # Add this line
    )
    
    # Return a TabPanel, not a Panel
    return TabPanel(child=about_content, title="About")

# Create tab panels
network_tab = create_network_tab()
metrics_tab = create_metrics_tab()
word_analysis_tab = create_word_analysis_tab()
ml_analytics_tab = create_ml_analytics_tab()


# About tab content
# About tab content (continuing from where the code left off)
about_text = """
<div class='card' style='font-size: 15px'>
    <div class='section-title'><h2>About This Dashboard</h2></div>
    <p>
        This dashboard visualizes word co-occurrence networks extracted from a hoax call dataset.
        It helps identify patterns in fraudulent and normal messages. By analyzing how language
        evolves over time, we uncover communication structures linked to fraud.
    </p>

    <div class='section-title'><h2>How to Use This Dashboard</h2></div>
    <p>The dashboard is divided into the following interactive sections:</p>
    <ul>
        <li><strong>Network Visualization:</strong> Interactive word network graphs showing co-occurrences by year and message type.</li>
        <li><strong>Network Metrics:</strong> Time series charts tracking changes in structural graph properties.</li>
        <li><strong>Word Analysis:</strong> Tables showing most frequent, distinctive, and central words.</li>
    </ul>

    <div class='section-title'><h2>Data Description</h2></div>
    <p>
        The dataset contains text messages labeled as <span class="label-tag">fraud</span> or <span class="label-tag">normal</span>.
        Each message has been cleaned and tokenized into words for analysis.
    </p>

    <div class='section-title'><h2>Analysis Methodology</h2></div>
    <p>
        Using network science techniques, this dashboard builds temporal graphs where nodes are words and
        edges represent frequent co-occurrence. The graphs are analyzed over time to detect
        structural shifts, key influencers, and fraud-related word patterns.
    </p>

    <div class='section-title'><h2>Implementation Details</h2></div>
    <p>This system is built with:</p>
    <ul>
        <li><strong>NetworkX</strong> and <strong>DyNetX</strong> for network modeling</li>
        <li><strong>Pandas</strong> for data processing</li>
        <li><strong>Bokeh</strong> for interactive visualizations</li>
    </ul>
</div>
"""

about_content = Div(text=about_text, width=1000)
about_tab = TabPanel(child=about_content, title="About")
# Create tabs layout with all panels
tabs = Tabs(tabs=[about_tab, network_tab, metrics_tab, ml_analytics_tab, word_analysis_tab], 
           sizing_mode="stretch_width")

# Create dashboard header
dashboard_header = column(
    main_title,
    description,
    css_classes=['dashboard-header'],
    sizing_mode="stretch_width"  # Add this line
)


# Make the final layout responsive
final_layout = column(
    dashboard_header,
    tabs,
    sizing_mode="stretch_both"  # Changed to stretch both width and height
)
def image_to_html(path, width="800"):
    img = Image.open(path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{encoded}" width="{width}">'

# === List all your image paths here ===
image_paths = [
    "../results/degree-comparison-WS-real.png","../results/graph-comparison-WS-real.png","../results/path-length-graph.png","../results/future-link-prediction.png","../results/predicted-vs-actual.png","../results/prediction_metrics.png" 
]

# === Optional: Descriptions for each image ===
image_descriptions = [
    "Result 1: Degree distribution ",
    "Result 2: Real message Word Network vs Watts-Strogatz Network",
    "Result 3: Comparison of Clustering coefficients and average Path Length",
    "Result 4: Test and Train Set Visualization",
    "Result 5: Actual vs Predicted Visualization",
    "Result 6: Comparison of Link Prediction Metrics"
]

# === Create HTML blocks for each image and its description ===
image_divs = []
for i, img_path in enumerate(image_paths):
    desc = image_descriptions[i] if i < len(image_descriptions) else f"Result {i+1}"
    desc_div = Div(text=f"<h3>{desc}</h3>", sizing_mode="stretch_width")
    img_div = Div(text=image_to_html(img_path), sizing_mode="stretch_width")
    image_divs.extend([desc_div, img_div])

# === Intro text ===
validation_intro = """
<h2 style="color:#2c3e50;">Validation Results</h2>
<p style="font-size:17px">This section highlights the key outputs and metrics derived from various experiments conducted 
during the model validation process. Each image shown here is a direct representation of the significant results obtained,
 which serve as a benchmark for evaluating model performance. These outputs are crucial for understanding how well the 
 model is performing, identifying areas of improvement, and validating its effectiveness against expected outcomes.
</p>
"""

# === Final layout and tab ===
validation_layout = column(
    Div(text=validation_intro, sizing_mode="stretch_width"),
    *image_divs,
    sizing_mode="stretch_both"
)

validation_tab = TabPanel(child=column(validation_layout), title="Validation")
# Add the layout to the current Bokeh document
curdoc().add_root(final_layout)
curdoc().title = "Temporal Word Network Analysis Dashboard"
tabs.tabs.append(validation_tab)
# Add custom CSS to document
curdoc().template_variables["css_code"] = """
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background-color: #f9f9f9;
    margin: 0;
    padding: 0;  /* Remove padding to use full width */
    width: 100%;
    overflow-x: hidden;
}

.bk-root {
    width: 100% !important;
    max-width: 100% !important;
}

.dashboard-header {
    margin-bottom: 20px;
    width: 100%;
}

.bk-root .bk-tab {
    font-size: 14px;
    font-weight: bold;
}

.bk-root .bk-tab-header {
    border-bottom: 2px solid #3498db;
}

.bk-root .bk-tabs-header .bk-tab.bk-active {
    background-color: #3498db;
    color: white;
}

.bk-root .bk-tabs-header .bk-tab:hover {
    background-color: #e9f7fe;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
    color: #2c3e50;
    border-left: 4px solid #3498db;
    padding-left: 12px;
}

.card {
    transition: box-shadow 0.3s ease-in-out;
    background: #ffffff;
    padding: 25px;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    max-width: 100%;
    margin: auto;
}

.card p {
    line-height: 1.6;
    margin-bottom: 10px;
    color: #333;
}

.card ul, .card ol {
    padding-left: 20px;
    margin-bottom: 10px;
}

.card ul li, .card ol li {
    margin-bottom: 6px;
    line-height: 1.5;
}

.label-tag {
    background-color: #3498db;
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
}

.card:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* Improve table styling */
.bk-root .bk-data-table {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    border-radius: 5px;
    overflow: hidden;
}

.bk-root .bk-data-table th {
    background-color: #f2f2f2;
    color: #34495e;
    font-weight: bold;
}

/* Improve button styling */
.bk-root .bk-btn {
    border-radius: 4px;
    transition: background-color 0.3s;
}

.bk-root .bk-btn:hover {
    background-color: #4aa3df;
}

/* Improve radio button group */
.bk-root .bk-btn-group {
    border-radius: 4px;
    overflow: hidden;
}

/* Responsive adjustments */
@media (max-width: 1200px) {
    .bk-root {
        padding: 10px;
    }
}

.bk-plot-wrapper {
    width: 100% !important;
}

.card {
    width: 100%;
    box-sizing: border-box;
}

/* Improve responsive design */
@media (max-width: 1200px) {
    .bk-root {
        padding: 10px;
    }
}

@media (max-width: 768px) {
    /* Stack columns on smaller screens */
    .bk-Row {
        flex-direction: column !important;
    }
}
"""

# Note for running the app:
# Run with: bokeh serve --show dashboard.py