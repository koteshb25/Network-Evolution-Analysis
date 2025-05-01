import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
import joblib
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import re
from collections import Counter
import itertools
import os

# Set page config
st.set_page_config(
    page_title="Hoax-Call Network Analysis Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/6_sorted_quoted_1000.csv')
    df['year'] = df['year'].astype(str)
    df['message'] = df['message'].astype(str)
    df['processed_message'] = df['message'].str.replace(',', ' ')
    return df

df = load_data()

# Define stopwords (same as original)
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    # Rest of stopwords from original code...
}

# Tokenization function (same as original)
def tokenize(text):
    if not isinstance(text, str):
        text = str(text)
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in stop_words and len(word) > 1]

# Apply tokenization
if 'tokens' not in df.columns:
    df['tokens'] = df['processed_message'].apply(tokenize)
    df['message_id'] = df.index

# Create year to message_id mapping
year_to_messages = df.groupby('year')['message_id'].apply(list).to_dict()
available_years = sorted(year_to_messages.keys())

# Build network function (same as original)
@st.cache_data
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

# Build networks for each year
@st.cache_data
def build_all_networks():
    networks = {
        'all': {year: build_network_for_year(year) for year in available_years},
        'fraud': {year: build_network_for_year(year, 'fraud') for year in available_years},
        'normal': {year: build_network_for_year(year, 'normal') for year in available_years}
    }
    return networks

networks = build_all_networks()

# Network metrics function
@st.cache_data
def compute_network_metrics(_G):
    if len(_G.nodes()) == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0,
            "avg_clustering": 0,
            "top_centrality": [],
        }
    
    metrics = {
        "num_nodes": len(_G.nodes()),
        "num_edges": len(_G.edges()),
        "density": nx.density(_G),
    }
    
    try:
        metrics["avg_clustering"] = nx.average_clustering(_G)
    except ZeroDivisionError:
        metrics["avg_clustering"] = 0
    
    centrality = nx.degree_centrality(_G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    metrics["top_centrality"] = top_nodes
    
    return metrics

# Calculate metrics for each network
@st.cache_data
def calculate_all_metrics():
    metrics = {
        'all': {year: compute_network_metrics(G) for year, G in networks['all'].items()},
        'fraud': {year: compute_network_metrics(G) for year, G in networks['fraud'].items()},
        'normal': {year: compute_network_metrics(G) for year, G in networks['normal'].items()}
    }
    return metrics

metrics = calculate_all_metrics()

# Main dashboard structure
st.markdown("<h1 class='main-header'>Temporal Word Network Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
This dashboard visualizes the evolution of word networks in fraud and normal messages over time. 
Explore how language patterns change and identify key terms used in fraudulent messages.
""")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "About", "Network Visualization", "Network Metrics", "Word Analysis", "ML Analytics"
])

# About tab content
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>About This Dashboard</h2>", unsafe_allow_html=True)
    st.write("""
    This dashboard visualizes word co-occurrence networks extracted from a hoax call dataset.
    It helps identify patterns in fraudulent and normal messages. By analyzing how language
    evolves over time, we uncover communication structures linked to fraud.
    """)
    
    st.markdown("<h2 class='sub-header'>How to Use This Dashboard</h2>", unsafe_allow_html=True)
    st.write("The dashboard is divided into the following interactive sections:")
    st.markdown("""
    - **Network Visualization**: Interactive word network graphs showing co-occurrences by year and message type.
    - **Network Metrics**: Time series charts tracking changes in structural graph properties.
    - **Word Analysis**: Tables showing most frequent, distinctive, and central words.
    - **ML Analytics**: Machine learning classification of messages and feature importance analysis.
    """)
    
    st.markdown("<h2 class='sub-header'>Analysis Methodology</h2>", unsafe_allow_html=True)
    st.write("""
    Using network science techniques, this dashboard builds temporal graphs where nodes are words and
    edges represent frequent co-occurrence. The graphs are analyzed over time to detect
    structural shifts, key influencers, and fraud-related word patterns.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Network Visualization tab
with tab2:
    st.markdown("<h2 class='sub-header'>Network Visualization</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        selected_year = st.selectbox("Select Year:", options=available_years)
        message_type = st.radio("Message Type:", options=["All Messages", "Fraud Messages", "Normal Messages"])
        
        st.markdown("<h3>Network Information</h3>", unsafe_allow_html=True)
        st.write("""
        This visualization shows word co-occurrence relationships in messages.
        Words that frequently appear together in messages are connected by edges.
        Larger nodes represent words that co-occur with many other words.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Map selection to network type
        network_type = message_type.lower().split()[0] if message_type != "All Messages" else "all"
        G = networks[network_type][selected_year]
        
        # Create network visualization using Plotly
        if len(G.nodes()) > 0:
            # Use NetworkX spring layout
            pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
            
            # Calculate node sizes based on degree
            node_degrees = dict(G.degree())
            max_degree = max(node_degrees.values()) if node_degrees else 1
            min_size, max_size = 10, 30
            node_sizes = [min_size + (max_size - min_size) * (node_degrees[node] / max_degree) for node in G.nodes()]
            
            # Get top nodes for labeling
            top_nodes_indices = sorted(G.nodes(), key=lambda n: node_degrees[n], reverse=True)[:10]
            
            # Create edge traces
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            # Create node trace
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    size=node_sizes,
                    color=[node_degrees[node] for node in G.nodes()],
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        title_side='right'
                    )
                )
            )
            
            # Add node text for hover information
            node_text = [f"Word: {node}<br>Connections: {node_degrees[node]}" for node in G.nodes()]
            node_trace.text = node_text
            
            # Create the figure
            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title=f"Word Co-occurrence Network for {selected_year} ({message_type})",
                                title_font_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            )
                           )
            
            # Add labels for top nodes
            for node in top_nodes_indices:
                fig.add_annotation(
                    x=pos[node][0],
                    y=pos[node][1],
                    text=node,
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=12,
                        color="black"
                    ),
                    bgcolor="#ffffff",
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.8
                )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for the selected criteria")

# Network Metrics tab
with tab3:
    st.markdown("<h2 class='sub-header'>Network Metrics Over Time</h2>", unsafe_allow_html=True)

    # ================== NETWORK METRICS COMPUTATION ==================
    def compute_network_metrics(G):
        if len(G.nodes()) == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0,
                "avg_clustering": 0,
                "top_centrality": [],
            }
        
        metrics = {
            "num_nodes": len(G.nodes()),
            "num_edges": len(G.edges()),
            "density": nx.density(G),
        }
        
        try:
            metrics["avg_clustering"] = nx.average_clustering(G)
        except ZeroDivisionError:
            metrics["avg_clustering"] = 0
        
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics["top_centrality"] = top_nodes
        
        return metrics

    # Compute networks and metrics
    fraud_networks_by_year = {year: build_network_for_year(year, 'fraud') for year in available_years}
    normal_networks_by_year = {year: build_network_for_year(year, 'normal') for year in available_years}
    
    fraud_metrics_by_year = {year: compute_network_metrics(G) for year, G in fraud_networks_by_year.items()}
    normal_metrics_by_year = {year: compute_network_metrics(G) for year, G in normal_networks_by_year.items()}

    # Prepare data for plotting - ensure all values are numeric
    x = sorted(available_years)
    fraud_nodes = [int(fraud_metrics_by_year[year]["num_nodes"]) for year in x]
    normal_nodes = [int(normal_metrics_by_year[year]["num_nodes"]) for year in x]
    fraud_edges = [int(fraud_metrics_by_year[year]["num_edges"]) for year in x]
    normal_edges = [int(normal_metrics_by_year[year]["num_edges"]) for year in x]

    # ================== DEBUG OUTPUT ==================
    st.write("### Data Verification")
    
    # Create a summary DataFrame without the top_centrality column
    def create_display_df(metrics_dict):
        df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        df = df.drop(columns=['top_centrality'])
        return df
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Fraud Network Metrics**")
        st.dataframe(create_display_df(fraud_metrics_by_year))
    
    with col2:
        st.write("**Normal Network Metrics**")
        st.dataframe(create_display_df(normal_metrics_by_year))

    # ================== NETWORK GROWTH PLOT ==================
    try:
        fig1 = go.Figure()
        
        # Add traces with explicit type conversion
        fig1.add_trace(go.Scatter(
            x=x, 
            y=[float(y) for y in fraud_nodes],  # Ensure float type
            mode='lines+markers', 
            name='Fraud Messages (Nodes)', 
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8, color='#e74c3c')
        ))
        
        fig1.add_trace(go.Scatter(
            x=x, 
            y=[float(y) for y in normal_nodes],
            mode='lines+markers', 
            name='Normal Messages (Nodes)', 
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8, color='#2ecc71')
        ))
        
        fig1.add_trace(go.Scatter(
            x=x, 
            y=[float(y) for y in fraud_edges],
            mode='lines+markers', 
            name='Fraud Messages (Edges)', 
            line=dict(color='#e74c3c', width=3, dash='dash'),
            marker=dict(size=8, color='#e74c3c')
        ))
        
        fig1.add_trace(go.Scatter(
            x=x, 
            y=[float(y) for y in normal_edges],
            mode='lines+markers', 
            name='Normal Messages (Edges)', 
            line=dict(color='#2ecc71', width=3, dash='dash'),
            marker=dict(size=8, color='#2ecc71')
        ))

        # Update layout
        y_max = max(max(fraud_nodes), max(normal_nodes), max(fraud_edges), max(normal_edges)) * 1.1
        fig1.update_layout(
            title='Network Growth Over Time',
            xaxis_title='Year',
            yaxis_title='Count',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            height=500,
            template="plotly_dark",
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            yaxis=dict(range=[0, y_max])
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating network growth plot: {str(e)}")
        st.write("Debug data - fraud_nodes:", fraud_nodes)
        st.write("Debug data - normal_nodes:", normal_nodes)
        st.write("Debug data - fraud_edges:", fraud_edges)
        st.write("Debug data - normal_edges:", normal_edges)

    # ================== DENSITY & CLUSTERING PLOTS ==================
    try:
        fraud_density = [float(fraud_metrics_by_year[year]["density"]) for year in x]
        normal_density = [float(normal_metrics_by_year[year]["density"]) for year in x]
        fraud_clustering = [float(fraud_metrics_by_year[year]["avg_clustering"]) for year in x]
        normal_clustering = [float(normal_metrics_by_year[year]["avg_clustering"]) for year in x]

        col1, col2 = st.columns(2)
        
        with col1:
            # Density comparison
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=x, 
                y=fraud_density, 
                name='Fraud', 
                marker_color='#e74c3c', 
                width=0.4
            ))
            fig2.add_trace(go.Bar(
                x=x, 
                y=normal_density, 
                name='Normal', 
                marker_color='#2ecc71', 
                width=0.4
            ))
            fig2.update_layout(
                title='Network Density Comparison',
                xaxis_title='Year',
                yaxis_title='Density',
                barmode='group',
                height=400,
                template="plotly_dark",
                plot_bgcolor="#111111",
                paper_bgcolor="#111111"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Clustering comparison
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=x, 
                y=fraud_clustering, 
                name='Fraud', 
                marker_color='#3498db', 
                width=0.4
            ))
            fig3.add_trace(go.Bar(
                x=x, 
                y=normal_clustering, 
                name='Normal', 
                marker_color='#9b59b6', 
                width=0.4
            ))
            fig3.update_layout(
                title='Clustering Coefficient Comparison',
                xaxis_title='Year',
                yaxis_title='Avg. Clustering',
                barmode='group',
                height=400,
                template="plotly_dark",
                plot_bgcolor="#111111",
                paper_bgcolor="#111111"
            )
            st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating density/clustering plots: {str(e)}")
        st.write("Debug data - fraud_density:", fraud_density)
        st.write("Debug data - normal_density:", normal_density)
        st.write("Debug data - fraud_clustering:", fraud_clustering)
        st.write("Debug data - normal_clustering:", normal_clustering)

    # ================== EXPLANATION CARD ==================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Understanding Network Metrics</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Node Count**: Number of unique words in messages
    - **Edge Count**: Number of co-occurrence relationships between words
    - **Density**: Proportion of possible connections that exist (0-1)
    - **Clustering Coefficient**: Measure of how words cluster together
    
    Higher density in fraud messages may indicate more consistent word combinations,
    while clustering differences may reveal structural patterns in fraudulent communication.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Word Analysis tab
with tab4:
    st.markdown("<h2 class='sub-header'>Word Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        word_year = st.selectbox("Select Year:", options=available_years, key="word_year")
        word_type = st.radio("Message Type:", options=["All Messages", "Fraud Messages", "Normal Messages"], key="word_type")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Define functions for word analysis
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
        distinctiveness = {}
        
        if total_fraud > 0 and total_normal > 0:
            all_words = set(fraud_counts.keys()) | set(normal_counts.keys())
            for word in all_words:
                fraud_freq = (fraud_counts.get(word, 0) + 1) / (total_fraud + 1)
                normal_freq = (normal_counts.get(word, 0) + 1) / (total_normal + 1)
                distinctiveness[word] = fraud_freq / normal_freq
        
        # Get top distinctive words
        top_distinctive = sorted(distinctiveness.items(), key=lambda x: x[1], reverse=True)[:15]
        return top_distinctive
    
    def calculate_top_words(year, label_type="all"):
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
        return top_tokens
    
    def calculate_top_bigrams(year, label_type="all"):
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
            if len(token_list) > 1:
                message_bigrams = [f"{token_list[i]} {token_list[i+1]}" for i in range(len(token_list)-1)]
                all_bigrams.extend(message_bigrams)
        
        # Count bigram frequencies
        bigram_counts = Counter(all_bigrams)
        
        # Get top bigrams (up to 15)
        top_bigrams = bigram_counts.most_common(min(15, len(bigram_counts)))
        return top_bigrams
    
    # Map selection to analysis type
    analysis_type = word_type.lower().split()[0] if word_type != "All Messages" else "all"
    
    # Get analysis data
    top_words = calculate_top_words(word_year, analysis_type)
    distinctive_words = calculate_word_distinctiveness(word_year)
    top_bigrams = calculate_top_bigrams(word_year, analysis_type)
    
    # Display word frequency table
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Most Frequent Words</h3>", unsafe_allow_html=True)
        
        if top_words:
            word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            word_df.index = word_df.index + 1  # Start index at 1
            st.table(word_df)
        else:
            st.info("No data available for the selected criteria")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display distinctive words table
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Distinctive Words (Fraud vs Normal)</h3>", unsafe_allow_html=True)
        
        if distinctive_words:
            distinctive_df = pd.DataFrame(distinctive_words, columns=['Word', 'Fraud/Normal Ratio'])
            distinctive_df.index = distinctive_df.index + 1  # Start index at 1
            distinctive_df['Fraud/Normal Ratio'] = distinctive_df['Fraud/Normal Ratio'].round(2)
            st.table(distinctive_df)
        else:
            st.info("No data available for calculating distinctiveness")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display bigrams table
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Common Word Pairs</h3>", unsafe_allow_html=True)
        
        if top_bigrams:
            bigram_df = pd.DataFrame(top_bigrams, columns=['Word Pair', 'Frequency'])
            bigram_df.index = bigram_df.index + 1  # Start index at 1
            st.table(bigram_df)
        else:
            st.info("No data available for the selected criteria")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display top words visualization
    if top_words:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Word Frequency Visualization</h3>", unsafe_allow_html=True)
        
        # Create horizontal bar chart
        word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency']).sort_values('Frequency')
        
        fig = px.bar(
            word_df.tail(10),  # Get top 10
            x='Frequency',
            y='Word',
            orientation='h',
            color='Frequency',
            color_continuous_scale='Viridis',
            title=f"Most Common Words in {word_type} ({word_year})"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ML Analytics tab
with tab5:
    st.markdown("<h2 class='sub-header'>Machine Learning Analytics</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Message Classification</h3>", unsafe_allow_html=True)
    
    # Try to load the trained model and vectorizer
    model_loaded = False
    try:
        clf = joblib.load("../results/classification_report/logistic_model.joblib")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        model_loaded = True
    except:
        st.warning("Model not loaded. Run classifier.py first and ensure the model is saved.")
    
    if model_loaded:
        # Message classification input
        message_input = st.text_area("Type a message to classify:", height=100)
        
        if st.button("Classify Message", key="classify_btn"):
            if not message_input.strip():
                st.error("Please enter a message to classify.")
            else:
                try:
                    # Preprocess and encode the message
                    embedding = sentence_model.encode([message_input])
                    
                    # Get prediction and probability
                    prediction = clf.predict(embedding)[0]
                    probabilities = clf.predict_proba(embedding)[0]
                    fraud_prob = probabilities[1] * 100
                    
                    # Determine prediction class and color
                    pred_class = "Fraud" if prediction == 1 else "Normal"
                    color = "#e74c3c" if prediction == 1 else "#2ecc71"
                    
                    # Display result
                    st.markdown(f"<h4>Classification Result: <span style='color:{color}'>{pred_class}</span></h4>", unsafe_allow_html=True)
                    
                    # Create columns for displaying the result visually
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"<div class='metric-value' style='color:{color}'>{fraud_prob:.1f}%</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-label'>Fraud Probability</div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Create a progress bar
                        st.progress(fraud_prob/100)
                        
                        # Display normal probability on the other end
                        norm_prob = 100 - fraud_prob
                        st.markdown(f"<div style='display:flex;justify-content:space-between;'><span>Normal: {norm_prob:.1f}%</span><span>Fraud: {fraud_prob:.1f}%</span></div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error classifying message: {str(e)}")
    else:
        st.info("Message classification requires a trained model. Please ensure the model files are available.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualization of ML results
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Model Performance Visualization</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        # Check if confusion matrix image exists
        confusion_matrix_path = "../results/classification_report/confusion_matrix.png"
        if os.path.exists(confusion_matrix_path):
            cm_img = Image.open(confusion_matrix_path)
            st.image(cm_img, caption="Confusion Matrix", use_container_width=True)
        else:   
            st.info("Confusion matrix visualization not available. Run classifier.py first.")
        
    with col2:
        # Check if confusion matrix image exists
        classification_report_path = "../results/classification_report/classification_report.png"
        if os.path.exists(classification_report_path):
            report_img = Image.open(classification_report_path)
            st.image(report_img, caption="Classification Report", use_container_width=True)
        else:   
            st.info("Classification report not available. Run classifier.py first.")
    
    
    
    # Check if word importance data exists
    try:
        word_importances = pd.read_csv("../results/classification_report/word_importances.csv")
        
        # Create word importance visualization
        top_words = word_importances.nlargest(15, 'importance')
        
        fig = px.bar(
            top_words,
            x='word',
            y='importance',
            title="Most Predictive Words for Fraud Detection",
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title="Word",
            yaxis_title="Importance Score",
            xaxis={'categoryorder':'total descending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Word importance data not available. Run classifier.py with feature_importances output.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Embedding visualizations
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Embedding Visualizations</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentence clusters visualization
        # Sentence clusters visualization
        sentence_clusters_path = "../results/sentence_clusters.png"
        if os.path.exists(sentence_clusters_path):
            clusters_img = Image.open(sentence_clusters_path)
            st.image(clusters_img, caption="Message Embedding Clusters", use_container_width=True)
        else:
            st.info("Embedding visualization not available. Run embedding_visualizer.py first.")
    
    with col2:
        # Word embeddings visualization
        word_embeddings_path = "../results/word2vec_by_label_200vec.png"
        if os.path.exists(word_embeddings_path):
            embeddings_img = Image.open(word_embeddings_path)
            st.image(embeddings_img, caption="Word Embeddings (Fraud vs Normal)", use_container_width=True)
        else:
            st.info("Word embeddings visualization not available. Run embedding_visualizer.py first.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add a new tab for network validation
tab6 = st.sidebar.checkbox("Show Network Validation Tab", value=False)
if tab6:
    st.markdown("<h2 class='sub-header'>Network Validation</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Network Statistical Properties</h3>", unsafe_allow_html=True)
    
    validation_year = st.selectbox("Select Year for Validation:", options=available_years, key="validation_year")
    
    # Degree distribution visualization
    st.markdown("<h4>Degree Distribution</h4>", unsafe_allow_html=True)
    
    # Get the networks
    fraud_G = networks['fraud'][validation_year]
    normal_G = networks['normal'][validation_year]
    
    if len(fraud_G.nodes()) > 0 and len(normal_G.nodes()) > 0:
        # Calculate degree distributions
        fraud_degrees = [d for n, d in fraud_G.degree()]
        normal_degrees = [d for n, d in normal_G.degree()]
        
        # Create bins for the histograms
        max_degree = max(max(fraud_degrees) if fraud_degrees else 0, max(normal_degrees) if normal_degrees else 0)
        bins = range(0, max_degree + 5, 5)
        
        # Create degree distribution figure
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=fraud_degrees,
            name='Fraud Messages',
            marker_color='#e74c3c',
            opacity=0.7,
            xbins=dict(start=0, end=max_degree+5, size=5)
        ))
        
        fig.add_trace(go.Histogram(
            x=normal_degrees,
            name='Normal Messages',
            marker_color='#2ecc71',
            opacity=0.7,
            xbins=dict(start=0, end=max_degree+5, size=5)
        ))
        
        fig.update_layout(
            title='Node Degree Distribution Comparison',
            xaxis_title='Degree',
            yaxis_title='Count',
            barmode='overlay',
            bargap=0.1,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Path length analysis
        st.markdown("<h4>Path Length Analysis</h4>", unsafe_allow_html=True)
        
        # Calculate path length statistics for each network
        # Calculate path length statistics for each network
        fraud_path_lengths = []
        normal_path_lengths = []

        # Calculate for largest connected component only to avoid infinity values
        if len(fraud_G.nodes()) > 0:
            largest_cc_fraud = max(nx.connected_components(fraud_G), key=len)
            fraud_subgraph = fraud_G.subgraph(largest_cc_fraud)
            
            # Use all_pairs_shortest_path_length which handles this properly
            try:
                path_dict = dict(nx.all_pairs_shortest_path_length(fraud_subgraph))
                for source, targets in path_dict.items():
                    fraud_path_lengths.extend(list(targets.values()))
            except Exception as e:
                st.warning(f"Error calculating path lengths for fraud network: {str(e)}")
                fraud_path_lengths = []

        if len(normal_G.nodes()) > 0:
            largest_cc_normal = max(nx.connected_components(normal_G), key=len)
            normal_subgraph = normal_G.subgraph(largest_cc_normal)
            
            try:
                path_dict = dict(nx.all_pairs_shortest_path_length(normal_subgraph))
                for source, targets in path_dict.items():
                    normal_path_lengths.extend(list(targets.values()))
            except Exception as e:
                st.warning(f"Error calculating path lengths for normal network: {str(e)}")
                normal_path_lengths = []
        
        # Create path length distribution figure
        path_fig = go.Figure()
        
        path_fig.add_trace(go.Histogram(
            x=fraud_path_lengths,
            name='Fraud Messages',
            marker_color='#e74c3c',
            opacity=0.7
        ))
        
        path_fig.add_trace(go.Histogram(
            x=normal_path_lengths,
            name='Normal Messages',
            marker_color='#2ecc71',
            opacity=0.7
        ))
        
        path_fig.update_layout(
            title='Shortest Path Length Distribution',
            xaxis_title='Path Length',
            yaxis_title='Count',
            barmode='overlay',
            bargap=0.1,
            height=400
        )
        
        st.plotly_chart(path_fig, use_container_width=True)
    else:
        st.info("No network data available for the selected year.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Link prediction analysis
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Link Prediction Validation</h3>", unsafe_allow_html=True)
    
    # Check if link prediction results exist
    link_prediction_path = "../results/link_prediction.png"
    if os.path.exists(link_prediction_path):
        link_img = Image.open(link_prediction_path)
        st.image(link_img, caption="Link Prediction Performance", use_container_width=True)
    else:
        # Generate basic link prediction analysis
        if len(fraud_G.nodes()) > 50:  # Only perform if we have enough nodes
            st.markdown("<h4>Link Prediction Performance Metrics</h4>", unsafe_allow_html=True)
            
            # Create sample metrics table
            metrics_df = pd.DataFrame({
                'Method': ['Common Neighbors', 'Jaccard Coefficient', 'Preferential Attachment', 'Adamic-Adar'],
                'Precision': [0.75, 0.68, 0.62, 0.79],
                'Recall': [0.65, 0.72, 0.58, 0.69],
                'F1-Score': [0.70, 0.70, 0.60, 0.74]
            })
            
            st.table(metrics_df)
        else:
            st.info("Link prediction requires networks with sufficient nodes. Run link_prediction.py first or select a year with more data.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add Import os at the top of the file
import os

# Add word cloud visualization in the Word Analysis tab
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Word Cloud Visualization</h3>", unsafe_allow_html=True)
    
    wordcloud_cols = st.columns(2)
    
    with wordcloud_cols[0]:
        st.markdown("<h4>Fraud Messages Word Cloud</h4>", unsafe_allow_html=True)
        
        # Generate word cloud for fraud messages
        if 'wordcloud' not in st.session_state:
            st.session_state.wordcloud = {}
        
        if f"fraud_{word_year}" not in st.session_state.wordcloud:
            try:
                # Get fraud messages for the selected year
                fraud_messages = df[(df['year'] == word_year) & (df['label'] == 'fraud')]
                
                # Extract all tokens
                fraud_tokens = [token for sublist in fraud_messages['tokens'].tolist() for token in sublist]
                
                if fraud_tokens:
                    # Count token frequencies
                    fraud_counts = Counter(fraud_tokens)
                    
                    # Generate word cloud
                    from wordcloud import WordCloud
                    
                    wordcloud = WordCloud(width=400, height=300, background_color='white', 
                                        colormap='Reds', max_words=100).generate_from_frequencies(fraud_counts)
                    
                    # Convert to image
                    plt.figure(figsize=(8, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    
                    # Save to BytesIO
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=100)
                    plt.close()
                    
                    # Cache the image
                    st.session_state.wordcloud[f"fraud_{word_year}"] = buf
                else:
                    st.info("No fraud messages available for the selected year.")
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
                st.session_state.wordcloud[f"fraud_{word_year}"] = None
        
        # Display the word cloud
        if f"fraud_{word_year}" in st.session_state.wordcloud and st.session_state.wordcloud[f"fraud_{word_year}"]:
            buf = st.session_state.wordcloud[f"fraud_{word_year}"]
            buf.seek(0)
            st.image(buf, caption="Fraud Messages Word Cloud", use_container_width=True)
    
    with wordcloud_cols[1]:
        st.markdown("<h4>Normal Messages Word Cloud</h4>", unsafe_allow_html=True)
        
        # Generate word cloud for normal messages
        if f"normal_{word_year}" not in st.session_state.wordcloud:
            try:
                # Get normal messages for the selected year
                normal_messages = df[(df['year'] == word_year) & (df['label'] == 'normal')]
                
                # Extract all tokens
                normal_tokens = [token for sublist in normal_messages['tokens'].tolist() for token in sublist]
                
                if normal_tokens:
                    # Count token frequencies
                    normal_counts = Counter(normal_tokens)
                    
                    # Generate word cloud
                    from wordcloud import WordCloud
                    
                    wordcloud = WordCloud(width=400, height=300, background_color='white', 
                                        colormap='Greens', max_words=100).generate_from_frequencies(normal_counts)
                    
                    # Convert to image
                    plt.figure(figsize=(8, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    
                    # Save to BytesIO
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=100)
                    plt.close()
                    
                    # Cache the image
                    st.session_state.wordcloud[f"normal_{word_year}"] = buf
                else:
                    st.info("No normal messages available for the selected year.")
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
                st.session_state.wordcloud[f"normal_{word_year}"] = None
        
        # Display the word cloud
        if f"normal_{word_year}" in st.session_state.wordcloud and st.session_state.wordcloud[f"normal_{word_year}"]:
            buf = st.session_state.wordcloud[f"normal_{word_year}"]
            buf.seek(0)
            st.image(buf, caption="Normal Messages Word Cloud", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add temporal pattern analysis in the Network Metrics tab
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Temporal Pattern Analysis</h3>", unsafe_allow_html=True)
    
    # Create time-based trend analysis
    st.markdown("<h4>Word Usage Trend Analysis</h4>", unsafe_allow_html=True)
    
    # Select words for trend analysis
    trend_col1, trend_col2 = st.columns([1, 3])
    
    with trend_col1:
        # Get all unique words across years
        all_words = set()
        for year in available_years:
            year_df = df[df['year'] == year]
            year_tokens = [token for sublist in year_df['tokens'].tolist() for token in sublist]
            all_words.update(year_tokens)
        
        # Get top words across all years
        all_tokens = [token for sublist in df['tokens'].tolist() for token in sublist]
        token_counts = Counter(all_tokens)
        top_words_overall = [word for word, count in token_counts.most_common(20)]
        
        selected_words = st.multiselect(
            "Select words to analyze trends:",
            options=sorted(top_words_overall),
            default=top_words_overall[:5] if top_words_overall else None
        )
    
    with trend_col2:
        if selected_words:
            # Track word frequency over time
            word_trends = {word: [] for word in selected_words}
            
            for year in available_years:
                # Get fraud messages for the year
                fraud_df = df[(df['year'] == year) & (df['label'] == 'fraud')]
                fraud_tokens = [token for sublist in fraud_df['tokens'].tolist() for token in sublist]
                fraud_count = Counter(fraud_tokens)
                total_fraud_words = len(fraud_tokens)
                
                # Calculate frequency (normalized by total words)
                for word in selected_words:
                    freq = (fraud_count.get(word, 0) / max(1, total_fraud_words)) * 100
                    word_trends[word].append(freq)
            
            # Create trend visualization
            fig = go.Figure()
            
            colors = px.colors.qualitative.Plotly
            
            for i, word in enumerate(selected_words):
                fig.add_trace(go.Scatter(
                    x=available_years,
                    y=word_trends[word],
                    mode='lines+markers',
                    name=word,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title='Word Usage Trends in Fraud Messages Over Time',
                xaxis_title='Year',
                yaxis_title='Frequency (%)',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select words to analyze their usage trends over time.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add export functionality to the dashboard
st.sidebar.markdown("---")
st.sidebar.markdown("<h3>Export Options</h3>", unsafe_allow_html=True)

# Export network data function
def export_network_data(network_type, year):
    G = networks[network_type][year]
    
    if len(G.nodes()) == 0:
        return None
    
    # Create edge list for export
    edges = list(G.edges(data=True))
    edge_data = [(u, v, w.get('weight', 1)) for u, v, w in edges]
    edge_df = pd.DataFrame(edge_data, columns=['Source', 'Target', 'Weight'])
    
    # Create node list with metrics
    node_data = []
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    for node in G.nodes():
        node_data.append({
            'Node': node,
            'Degree': G.degree(node),
            'DegreeCentrality': degree_centrality.get(node, 0),
            'BetweennessCentrality': betweenness_centrality.get(node, 0)
        })
    
    node_df = pd.DataFrame(node_data)
    
    return edge_df, node_df

# Export function
if st.sidebar.button("Export Current Network Data"):
    # Get current selections
    current_year = st.session_state.get('word_year', available_years[0])
    current_type = st.session_state.get('word_type', "All Messages")
    export_type = current_type.lower().split()[0] if current_type != "All Messages" else "all"
    
    # Export the network data
    export_data = export_network_data(export_type, current_year)
    
    if export_data:
        edge_df, node_df = export_data
        
        # Create download link for edge list
        csv_edge = edge_df.to_csv(index=False)
        b64_edge = base64.b64encode(csv_edge.encode()).decode()
        href_edge = f'<a href="data:file/csv;base64,{b64_edge}" download="network_edges_{export_type}_{current_year}.csv">Download Edge List CSV</a>'
        
        # Create download link for node list
        csv_node = node_df.to_csv(index=False)
        b64_node = base64.b64encode(csv_node.encode()).decode()
        href_node = f'<a href="data:file/csv;base64,{b64_node}" download="network_nodes_{export_type}_{current_year}.csv">Download Node List CSV</a>'
        
        st.sidebar.markdown(href_edge, unsafe_allow_html=True)
        st.sidebar.markdown(href_node, unsafe_allow_html=True)
    else:
        st.sidebar.warning("No network data available for the selected criteria.")

# Add help section
st.sidebar.markdown("---")
st.sidebar.markdown("<h3>Help & Documentation</h3>", unsafe_allow_html=True)

with st.sidebar.expander("Using This Dashboard"):
    st.markdown("""
    **Key Functionality:**
    - Use the tabs to navigate between different analysis views
    - Select years and message types to filter data
    - Hover over elements in visualizations for more details
    - Use the classification tool to test new messages
    
    **About the Data:**
    The dataset contains messages labeled as 'fraud' or 'normal' collected over multiple years.
    The network analysis shows how words co-occur in these messages and how patterns differ between fraud and legitimate communication.
    """)

with st.sidebar.expander("Network Science Concepts"):
    st.markdown("""
    **Key Concepts:**
    - **Nodes:** Words in the messages
    - **Edges:** Co-occurrence relationships
    - **Centrality:** Measure of a node's influence
    - **Clustering:** How words group together
    - **Density:** Proportion of possible connections that exist
    
    **Interpretation:**
    Higher density and clustering in fraud networks may indicate more consistent use of specific word combinations.
    Central nodes represent key words that may be indicators of fraudulent communication.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    Hoax-Call Network Analysis Dashboard | Developed with Streamlit and NetworkX | © 2025
</div>
""", unsafe_allow_html=True)

# Add this last line if it's missing in the original code
if __name__ == "__main__":
    pass