import pandas as pd
from bokeh.io import curdoc
from bokeh.plotting import figure, from_networkx
from bokeh.models import HoverTool, Slider, Div
from bokeh.layouts import column, row
from bokeh.transform import linear_cmap
import networkx as nx
from collections import defaultdict
import itertools

# run command : bokeh serve --show call_types_over_years.py

# Load and prepare data
df = pd.read_csv("../data/6_sorted_quoted.csv")
df = df.rename(columns={'label': 'type', 'message': 'text'})
df['year'] = pd.to_numeric(df['year'], errors='coerce').dropna().astype(int)
df['words'] = df['text'].str.split(',')
df['period'] = pd.cut(df['year'], bins=range(2011, 2027, 3), 
                     labels=[f"{y}-{y+2}" for y in range(2011, 2026, 3)],
                     right=False)

# 1. Network creation function with error handling
def create_cooccurrence_network(words_list, min_edges=2):
    cooc = defaultdict(int)
    for words in words_list:
        if not isinstance(words, list):
            continue  # skip invalid entries
        words = [w.strip() for w in words if isinstance(w, str) and w.strip()]
        for pair in itertools.combinations(set(words), 2):
            cooc[tuple(sorted(pair))] += 1
    
    G = nx.Graph()
    for (w1, w2), count in cooc.items():
        if count >= min_edges:
            G.add_edge(w1, w2, weight=count)
    
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

# 2. Create all possible networks upfront for performance
periods = sorted(df['period'].unique())
call_types = df['type'].unique()

networks = {}
for period in periods:
    networks[period] = {}
    for call_type in call_types:
        subset = df[(df['period'] == period) & (df['type'] == call_type)]
        if len(subset) > 0:
            networks[period][call_type] = create_cooccurrence_network(subset['words'].tolist())

# 3. Create main figures
trends_plot = figure(title="Call Type Trends Over Time", 
                    x_axis_label='Year', 
                    y_axis_label='Count',
                    x_range=(2011, 2025), 
                    width=800, height=400)

# Plot temporal trends
trends = df.groupby(['year', 'type']).size().unstack().fillna(0)
colors = {'fraud': 'red', 'normal': 'blue'}
for call_type in df['type'].unique():
    if call_type in trends.columns:
        trends_plot.line(trends.index, trends[call_type], 
                        line_width=2, color=colors.get(call_type, 'gray'), 
                        legend_label=call_type)

trends_plot.legend.location = "top_left"
trends_plot.add_tools(HoverTool(tooltips=[("Year", "@x"), ("Count", "@y")]))

# Create network plot
network_plot = figure(title="Call Network", 
                     x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                     tools="pan,wheel_zoom,box_zoom,reset,tap,save",
                     width=600, height=600)

# 4. Create controls
period_select = Slider(title="Time Period", 
                     start=0, end=len(periods)-1, 
                     step=1, value=0, 
                     width=300)

call_type_select = Slider(title=f"Call Type (0={call_types[0]}, 1={call_types[1]})", 
                         start=0, end=len(call_types)-1, 
                         step=1, value=0, 
                         width=300)

# 5. Update function for Bokeh server
def update_network(attr, old, new):
    # Get current selections
    period_idx = period_select.value
    type_idx = call_type_select.value
    
    period = periods[period_idx]
    call_type = call_types[type_idx]
    
    # Clear previous renderers
    network_plot.renderers = []
    network_plot.title.text = f"{call_type.capitalize()} Calls Network: {period}"
    
    # Check if data exists
    if period not in networks or call_type not in networks[period]:
        network_plot.title.text = f"No {call_type} data for {period}"
        return
    
    G = networks[period][call_type]
    
    if len(G.edges()) == 0:
        network_plot.title.text = f"No connections in {call_type} calls for {period}"
        return
    
    # Create new layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    graph_renderer = from_networkx(G, pos, scale=1, center=(0,0))
    
    # Style nodes
    graph_renderer.node_renderer.glyph.size = 15
    graph_renderer.node_renderer.glyph.fill_color = colors.get(call_type, 'gray')
    
    # Style edges
    # graph_renderer.edge_renderer.glyph.line_width = {'field': 'weight', 'transform': linear_cmap('weight', 'Greys', 1, 5)}
    
    # Add hover tool
    hover = HoverTool(tooltips=[("Word", "@index")])
    network_plot.add_tools(hover)
    
    # Add to plot
    network_plot.renderers.append(graph_renderer)

# Set up callbacks
period_select.on_change('value', update_network)
call_type_select.on_change('value', update_network)

# Initial update
update_network(None, None, None)

# 6. Create layout
controls = column(
    Div(text="<h3>Network Explorer</h3>"),
    period_select,
    call_type_select,
    Div(text="<p>Select time period and call type to explore</p>")
)

dashboard = column(
    trends_plot,
    row(controls, network_plot)
)

# 7. Add to document
curdoc().add_root(dashboard)
curdoc().title = "Hoax Call Network Analysis"