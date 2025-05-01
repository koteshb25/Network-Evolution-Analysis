# Network Evolution Analysis

## Assignment Overview

This project aims to build a **privacy-aware system** to analyze how hoax call networks evolve over time. Using **temporal network modeling**, **dynamic SNA metrics**, and **public datasets**, it helps detect structural changes, influencers, and suspicious trends. The system emphasizes **open-source tools** and **visual analytics** for **proactive threat detection**.

---

## Setup Instructions

1. Clone the repository:
    ```bash
   ''' git clone "https://github.com/VatsalBhuva11/hoax-call-temporal-analysis.git"
    cd hoax-call-temporal-analysis
    ```
2. Create a virtual environment (recommended via conda):

    ```bash
    python -m venv venv
    ```

    If on Linux:

    ```bash
    source ./venv/bin/activate
    ```

    For Windows:

    ```bash
    venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Description of Directory Structure

```
.
├── venv/                    # Virtual Environment for this project. Will be created after running the venv command above.
│   ├── bin/      # Virtual environment binaries (auto-generated)
│   ├── include/             # Virtual environment include files
│   ├── lib64/             # Symlink to 'lib/' on 64-bit systems
│   └── lib/                # Python libraries for the virtual environment
├── code/              # Core Python scripts for modeling, analysis, and visualization
├── data/              # Processed datasets used in analysis (e.g., cleaned CSVs)
├── results/           # Generated outputs like plots, metrics, and screenshots
│   └── classification_report/  # Results from message classification
├── .gitignore             # Git configuration to exclude unnecessary files
├── pyvenv.cfg             # Virtual environment configuration file
├── README.md              # Project overview, setup instructions, and documentation
├── Group_16_SNA.pptx              # A presentation on our work, covering key topics.
└── requirements.txt       # List of Python dependencies for installation
```

---

## Dependencies

-   Python 3.9+, Jupyter Notebook
-   NetworkX, DyNetX, graph-tool, igraph
-   ruptures, changepy, STUMPY, scikit-multiflow, pyts
-   matplotlib, plotly, bokeh
-   scikit-learn, sentence-transformers, gensim, UMAP

    Use `requirements.txt` to install.

    ```bash
    pip install -r requirements.txt
    ```

---

## Features Implemented

This project integrates several network analysis, visualization, and time-series techniques to study how hoax call word networks evolve over time. Below is a description of the core features and methodologies used:

### Community Detection Algorithm (Greedy Modularity)

We applied the **Greedy Modularity Community Detection Algorithm** from NetworkX to uncover clusters of words that tend to co-occur frequently. These clusters often reflect repeated templates or themes within fraudulent messages. This approach helps identify:

-   Tightly-connected word groups,
-   Structural patterns in message content,
-   Differentiation between fraud and normal communication.

### Network Visualization using NetworkX, Matplotlib, and Plotly

We visualized the structure of word co-occurrence networks using:

-   **NetworkX** for graph construction and analysis,
-   **Matplotlib** for basic static visualizations,
-   **Plotly** for interactive and zoomable network plots.

These visualizations enabled us to explore:

-   The connectivity between words,
-   Node importance based on centrality,
-   The differences in network structure between fraudulent and normal messages.

### Temporal Analysis using DyNetX

To understand how fraud strategies change over time, we employed **DyNetX** for temporal graph modeling. The dataset was segmented year-wise to build dynamic word networks. Temporal analysis helped us:

-   Track the evolution of message complexity and structure,
-   Detect growth or shrinkage in fraud-related vocabulary networks,
-   Identify periods of significant structural change in communication patterns.

### Web-Based Dashboard using Bokeh

An interactive dashboard was built using **Bokeh** to present:

-   Dynamic word network visualizations across years,
-   Time series of graph metrics (nodes, edges, density, etc.),
-   Tables of the most frequent and central words.

The dashboard allows filtering by year and message type (fraud or normal), making it easy to explore how the language used in fraud evolves over time.

### Change Detection using Ruptures

To detect abrupt structural changes in the temporal data, we used the **Ruptures** library for change point detection. Applied to time series of metrics such as edge count and average degree, this helped identify:

-   Years where the structure of the fraud network changed significantly,
-   Potential emergence of new fraud templates or messaging strategies.

### Classical Growth Model Simulation (Watts-Strogatz Model)

To understand the structural nature of fraud word networks, we simulated a **Watts–Strogatz (WS)** small-world model with the same number of nodes. This allowed us to:

-   Compare the clustering and connectivity of real and synthetic graphs,
-   Determine whether fraud messages exhibit small-world properties,
-   Highlight differences in message complexity and connectivity.

### Message Classification using Machine Learning

Using SentenceBERT embeddings and logistic regression, we built a classifier to automatically differentiate between fraudulent and normal messages:

-   **Sentence embeddings**: Messages are converted to vector representations using the all-MiniLM-L6-v2 model
-   **Balanced logistic regression**: Addresses class imbalance in the dataset
-   **Performance metrics**: Classification report with precision, recall, and F1-score
-   **Visualization**: Confusion matrix to understand model errors
-   **Feature analysis**: Word importance scores to identify key fraud indicators

This classification system enables automated filtering and analysis of incoming messages based on their similarity to known fraud patterns.

### Embedding Analysis for Pattern Recognition

We implemented advanced embedding techniques to understand the semantic patterns within messages:

-   **SentenceBERT embeddings**: To capture the overall meaning of messages
-   **Word2Vec embeddings**: To analyze word-level relationships and context
-   **Dimensionality reduction**: Using UMAP to visualize high-dimensional embeddings in 2D space
-   **Clustering**: K-means clustering to identify message groups with similar semantic content
-   **Word usage analysis**: Visualization of words by their dominant usage (fraud vs. normal)

These embedding techniques help reveal subtle linguistic patterns that distinguish fraudulent from normal messages and show how fraud techniques evolve over time.

### Centrality Measures and Statistical Metrics

Key network statistics were computed to evaluate the role and influence of individual words. Metrics included:

-   **Degree centrality**: Identifies the most frequently co-occurring words,
-   **Betweenness centrality**: Highlights words that connect different word clusters,
-   **Clustering coefficient**: Measures the tendency of word groups to form templates,
-   **Graph density and average degree**: Provide insight into the overall compactness and interconnectedness of the message network.

These measures were tracked over time to quantify changes in fraud messaging behavior.

---

## Instructions to Reproduce Results

1. Use `/code/community_detection.ipynb` to analyse communities within the word-network.
2. Use `code/watts_strogatz_growth_model.ipynb` to simulate network growth and obtain key conclusions.
3. Use `code/future-link-prediction-validation.ipynb` to check how the model can evolve in the future.
4. Run `python code/classifier.py` to train and evaluate the fraud message classifier.
5. Run `python code/embeddings.py` to generate word and sentence embeddings visualizations.
6. From the `code/` directory, use command `bokeh serve --show dashboard.py` to explore temporal graphs, word analytics, and metrics. This opens the main interactive dashboard.
7. From the `code/` directory, Use command `bokeh serve --show call_types_over_years.py` to see the trend of fraud and normal calls over the years.

---

## Sample Outputs / Screenshots

-   Temporal graph evolution screenshots (in `results/`)
-   Change detection plots: clustering coefficient over time
-   Community merging/splitting visualizations
-   Link prediction precision-recall plots
-   Bokeh dashboard outputs and walkthrough
-   Word network plotted using Networkx, DyNetx, Plotly, and Matplotlib
-   Computation of centrality measures such as degree centrality
-   Recurrence plot comparing edge counts over snapshots of years
-   Real Graph and WS Model Network comparisons
-   Classification confusion matrix and performance metrics (`results/classification_report/`)
-   Word embedding clusters showing fraud vs. normal message patterns (`results/word2vec_by_label.png`)
-   Sentence embedding visualizations showing cluster separation (`results/sentence_clusters.png`)

---

## Thank you!
