import streamlit as st
import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap
import json_parser as jsp
import tempfile
import time

st.set_page_config(page_title="Schema Mapping Visualization", layout="wide")

# Add custom CSS to increase sidebar text size
st.markdown("""
<style>
    /* Target the main sidebar content and text */
    [data-testid="stSidebar"] {
        font-size: 18px !important;
    }
    
    /* Target sidebar headers specifically */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 22px !important;
    }
    
    /* Target radio buttons labels */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div {
        font-size: 18px !important;
    }
    
    /* Target checkbox labels */
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stCheckbox div {
        font-size: 18px !important;
    }
    
    /* Target slider text */
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider div {
        font-size: 18px !important;
    }
    
    /* Target selectbox and multiselect */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        font-size: 18px !important;
    }
    
    /* File uploader label */
    [data-testid="stSidebar"] .stFileUploader label {
        font-size: 18px !important;
    }
    
    /* Target any buttons in the sidebar */
    [data-testid="stSidebar"] button {
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to create custom metric display with consistent font sizes
def custom_metric(label, value):
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px; border-radius: 5px;">
            <div style="font-size: 24px; font-weight: bold;">{label}</div>
            <div style="font-size: 36px; font-weight: bold;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model():
    """Load and cache the sentence transformer model"""
    with st.spinner("Loading embedding model... This may take a minute on first run."):
        return SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)

def process_sentences_to_tables(sentences):
    # print('The sententeces ',sentences)
    tables = {}
    for sentence in sentences:
        table_name = sentence.split('->')[0].strip()
        if table_name not in tables:
            tables[table_name] = []
        content = '->'.join(sentence.split('->')[1:])
        tables[table_name].append([x.lower().strip() for x in content.split('->')])
    # print(tables)
    return tables

def create_embeddings(table_data, table_name, model, numerical=0, embedding_method="concatenate"):
    embeddings = [] 
    if numerical == 1:
        for idx, item in enumerate(table_data):
            category, subcategory, data = item[0], item[1:-1], item[-1]
            combined_info_from_above = f'{item[0]} -> {item[1:-1]} -> {item[-1]}'
            emb_category = model.encode(category)
            emb_subcategory = 0
            for index, value in enumerate(subcategory):
                emb_subcategory += model.encode(value) * (len(subcategory) - (index)) / len(subcategory)
            emb_data = model.encode(data)
            
            if embedding_method == "concatenate":
                combined_emb = np.concatenate([emb_category, emb_subcategory, emb_data], axis=0)
            else:  # weighted average
                combined_emb = (emb_category*1.0 + emb_subcategory*0.66 + emb_data*0.33) / 3.0
                
            embeddings.append((table_name, category, '->'.join(subcategory), idx, combined_emb, combined_info_from_above))
    else:
        for idx, item in enumerate(table_data):
            category, subcategory, data = item[0], item[1:-1], item[-1]
            emb_category = model.encode(category)
            emb_subcategory = np.zeros_like(emb_category)
            for index, value in enumerate(subcategory):
                emb_subcategory += model.encode(value) * (len(subcategory) - (index)) / len(subcategory)
            emb_data = model.encode(data)
            
            if embedding_method == "concatenate":
                combined_emb = np.concatenate([emb_category, emb_subcategory, emb_data], axis=0)
            else:  # weighted average
                combined_emb = (emb_category + emb_subcategory + emb_data) / 3.0
                
            embeddings.append((table_name, category, '->'.join(subcategory), idx, combined_emb))
    return embeddings

def generate_color_palette(num_tables):
    """Generate a color palette for the tables"""
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if num_tables <= len(base_colors):
        return base_colors[:num_tables]
    else:
        return plt.cm.tab20(np.linspace(0, 1, num_tables))

def compute_base_embeddings(information_to_be_checked, model, numerical, embedding_method):
    """
    Compute the base embeddings once, to be reused with different dimensionality reduction settings.
    """
    # Process sentences into tables
    tables = process_sentences_to_tables(information_to_be_checked)

    # Generate embeddings for each table
    all_embeddings = []
    for table_name, table_data in tables.items():
        table_embeddings = create_embeddings(table_data, table_name, model, numerical, embedding_method)
        all_embeddings.extend(table_embeddings)

    # Prepare data for similarity computation
    keys = [(t[0], t[1], t[3]) for t in all_embeddings]
    embedding_matrix = [t[4] for t in all_embeddings]
    
    # Create node info
    nodes = []
    for t in all_embeddings:
        node = f"{t[1]}\n {t[2]}\n ({t[0]})"
        nodes.append((node, t[0]))  # node label and table name
    
    # Generate color palette
    color_map = dict(zip(tables.keys(), generate_color_palette(len(tables))))
    
    # Compute node positions
    x_spacing = 2
    y_spacing = 2
    pos = {}
    for table_idx, table_name in enumerate(tables.keys()):
        table_nodes = [node for node, table in nodes if table == table_name]
        for i, node in enumerate(table_nodes):
            pos[node] = (table_idx * x_spacing, i * y_spacing)
    
    return {
        'nodes': nodes,
        'embedding_matrix': embedding_matrix,
        'keys': keys,
        'pos': pos,
        'color_map': color_map,
        'tables': tables
    }

def apply_dimensionality_reduction(base_data, dim_reduction, n_components):
    """
    Apply dimensionality reduction to the precomputed embeddings.
    """
    embedding_matrix = base_data['embedding_matrix']
    nodes = base_data['nodes']
    keys = base_data['keys']
    pos = base_data['pos']
    color_map = base_data['color_map']
    tables = base_data['tables']
    
    reduced_matrix = embedding_matrix
    
    # Apply dimensionality reduction if specified
    if dim_reduction == "pca" and len(embedding_matrix) > n_components:
        try:
            pca = PCA(n_components=min(n_components, len(embedding_matrix)))
            reduced_matrix = pca.fit_transform(embedding_matrix)
        except Exception as e:
            st.warning(f"PCA dimensionality reduction failed: {str(e)}. Using original embeddings.")
    
    elif dim_reduction == "umap" and len(embedding_matrix) > 2:
        try:
            reducer = umap.UMAP(n_components=min(n_components, len(embedding_matrix)), 
                                metric='cosine', 
                                random_state=42)
            reduced_matrix = reducer.fit_transform(embedding_matrix)
        except Exception as e:
            st.warning(f"UMAP dimensionality reduction failed: {str(e)}. Using original embeddings.")

    # Compute similarity matrix
    sim_matrix = cosine_similarity(reduced_matrix)
    
    # Pre-compute all possible edges with their similarities
    all_possible_edges = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i][0] != keys[j][0]:  # Only connect nodes from different tables
                node_i = f"{nodes[i][0]}"
                node_j = f"{nodes[j][0]}"
                similarity = sim_matrix[i, j]
                all_possible_edges.append((node_i, node_j, similarity, keys[i][0]))
    
    return {
        'nodes': nodes,
        'all_possible_edges': all_possible_edges,
        'pos': pos,
        'color_map': color_map,
        'tables': tables
    }

def create_similarity_network(network_data, similarity_threshold):
    """
    Create a graph based on pre-computed network data and a threshold.
    """
    nodes = network_data['nodes']
    all_possible_edges = network_data['all_possible_edges']
    pos = network_data['pos']
    color_map = network_data['color_map']
    tables = network_data['tables']
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with colors
    for node, table_name in nodes:
        G.add_node(node, color=color_map[table_name])
    
    # Filter edges based on threshold
    filtered_edges = [(edge[0], edge[1]) for edge in all_possible_edges if edge[2] >= similarity_threshold]
    G.add_edges_from(filtered_edges)
    
    return G, pos, color_map, tables

def main():
    st.title("Schema Matching Visualization")
    
    st.sidebar.header("Upload and Configure")
    
    uploaded_file = st.sidebar.file_uploader("Upload file", type=["json"])
    mapper_type = st.sidebar.radio(
        "Mapper Type", 
        options=["HMD_mapper", "VMD_mapper"],
        help="Choose which mapping function to use"
    )
    
    numerical = st.sidebar.radio(
        "Data Type", 
        options=[0, 1], 
        format_func={0: "Non-numerical", 1: "Numerical"}.get,
        help="Choose 1 if your data is numerical, otherwise choose 0"
    )
    
    embedding_method = st.sidebar.radio(
        "Embedding Combination Method",
        options=["concatenate", "weighted_average"],
        format_func={"concatenate": "Concatenation", "weighted_average": "Weighted Average"}.get,
        help="Choose how to combine the embeddings of different parts of the data"
    )
    
    # Add dimensionality reduction options
    st.sidebar.header("Advanced Options")
    dim_reduction = st.sidebar.radio(
        "Dimensionality Reduction",
        options=[None, "pca", "umap"],
        format_func={None: "None", "pca": "PCA", "umap": "UMAP"}.get,
        help="Apply dimensionality reduction to embeddings before calculating similarity"
    )
    
    # Show additional options based on dimensionality reduction choice
    n_components = 10
    if dim_reduction:
        n_components = st.sidebar.slider(
            f"{dim_reduction.upper()} Components", 
            min_value=2, 
            max_value=50, 
            value=10,
            help="Number of dimensions to reduce to"
        )
    
    # Create a session state to store our computed data
    if 'base_embeddings' not in st.session_state:
        st.session_state.base_embeddings = None
    if 'network_data' not in st.session_state:
        st.session_state.network_data = None
    if 'information_to_be_checked' not in st.session_state:
        st.session_state.information_to_be_checked = None
    if 'selected_tables' not in st.session_state:
        st.session_state.selected_tables = []
    if 'needs_recompute_base' not in st.session_state:
        st.session_state.needs_recompute_base = True
    if 'needs_recompute_dim' not in st.session_state:
        st.session_state.needs_recompute_dim = True
    if 'embedding_method' not in st.session_state:
        st.session_state.embedding_method = None
    if 'numerical' not in st.session_state:
        st.session_state.numerical = None
    if 'mapper_type' not in st.session_state:
        st.session_state.mapper_type = None
    if 'dim_reduction' not in st.session_state:
        st.session_state.dim_reduction = None
    if 'n_components' not in st.session_state:
        st.session_state.n_components = None
    
    selected_tables = []
    if uploaded_file is not None:
        # Load the uploaded JSON file
        json_content = json.loads(uploaded_file.read())
        
        # Display available tables for selection
        available_tables = list(json_content.keys())
        if len(available_tables) > 0:
            st.sidebar.header("Select Tables")
            
            # Create a container for table selection to detect changes
            with st.sidebar:
                for table in available_tables:
                    selected = st.checkbox(f"Include {table}", key=table)
                    if selected:
                        selected_tables.append(table)
            
            # Check if base computation parameters have changed
            if (set(selected_tables) != set(st.session_state.selected_tables) or
                embedding_method != st.session_state.embedding_method or
                numerical != st.session_state.numerical or
                mapper_type != st.session_state.mapper_type):
                st.session_state.selected_tables = selected_tables
                st.session_state.embedding_method = embedding_method
                st.session_state.numerical = numerical
                st.session_state.mapper_type = mapper_type
                st.session_state.needs_recompute_base = True
                st.session_state.needs_recompute_dim = True
            
            # Check if only dimensionality reduction parameters have changed
            if (dim_reduction != st.session_state.dim_reduction or
                n_components != st.session_state.n_components):
                st.session_state.dim_reduction = dim_reduction
                st.session_state.n_components = n_components
                st.session_state.needs_recompute_dim = True
        
        if len(selected_tables) >= 2:
            # Set up the threshold slider
            st.sidebar.header("Adjust Threshold")
            threshold = st.sidebar.slider("Similarity Threshold", 
                                      min_value=0.80, 
                                      max_value=1.0, 
                                      value=0.85, 
                                      step=0.01,
                                      help="Adjust similarity threshold to filter connections")
            
            # Load the model (cached)
            model = load_model()
            
            # Compute base embeddings only when needed
            if st.session_state.needs_recompute_base:
                with st.spinner("Computing base embeddings... This may take a moment."):
                    # Filter the JSON content based on selected tables
                    filtered_json = {table: json_content[table] for table in selected_tables}
                    
                    # Process the filtered JSON with the selected mapper
                    mapper_func = jsp.HMD_mapper if mapper_type == "HMD_mapper" else jsp.VMD_mapper
                    information_to_be_checked = mapper_func(filtered_json)
                    st.session_state.information_to_be_checked = information_to_be_checked
                    
                    # Compute base embeddings (this is the expensive operation)
                    base_embeddings = compute_base_embeddings(
                        information_to_be_checked,
                        model,
                        numerical,
                        embedding_method
                    )
                    st.session_state.base_embeddings = base_embeddings
                    st.session_state.needs_recompute_base = False
                    st.session_state.needs_recompute_dim = True
            
            # Apply dimensionality reduction if base embeddings exist and dim settings changed
            if st.session_state.needs_recompute_dim and st.session_state.base_embeddings is not None:
                with st.spinner(f"Applying {dim_reduction if dim_reduction else 'no'} dimensionality reduction..."):
                    # Apply dimensionality reduction to base embeddings
                    network_data = apply_dimensionality_reduction(
                        st.session_state.base_embeddings,
                        dim_reduction,
                        n_components
                    )
                    st.session_state.network_data = network_data
                    st.session_state.needs_recompute_dim = False
            
            # Create graph based on threshold (fast operation)
            G, pos, color_map, tables = create_similarity_network(
                st.session_state.network_data,
                threshold
            )
            
            # Display metrics and configuration using custom metrics
            st.header("Network Statistics")
            
            # Replace st.metric with custom HTML metrics for consistent sizing
            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="margin-bottom: 5px; font-size: 28px;">Nodes</h2>
                    <h1 style="font-size: 36px; margin-top: 0;">{len(G.nodes())}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="margin-bottom: 5px; font-size: 28px;">Connections</h2>
                    <h1 style="font-size: 36px; margin-top: 0;">{len(G.edges())}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="margin-bottom: 5px; font-size: 28px;">Tables</h2>
                    <h1 style="font-size: 36px; margin-top: 0;">{len(tables)}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Display configuration info
            st.header("Analysis Configuration")
            config_cols = st.columns(3)
            with config_cols[0]:
                st.info(f"**Embedding Method:** {'Concatenation' if embedding_method == 'concatenate' else 'Weighted Average'}")
            with config_cols[1]:
                if dim_reduction:
                    st.info(f"**Dimensionality Reduction:** {dim_reduction.upper()} ({n_components} components)")
                else:
                    st.info("**Dimensionality Reduction:** None")
            with config_cols[2]:
                st.info(f"**Mapping Method:** {mapper_type}")
                
            # Add a performance note
            compute_status = ""
            if st.session_state.needs_recompute_base:
                compute_status = "Full recomputation needed"
            elif st.session_state.needs_recompute_dim:
                compute_status = "Applying dimensionality reduction only"
            else:
                compute_status = "Using cached computations"
                
            st.caption(f"Status: {compute_status} | Change tables or embedding method to recalculate base embeddings. Change dimensionality reduction to reapply on cached embeddings.")
            
            # Visualization settings
            st.sidebar.header("Visualization Settings")
            fig_width = st.sidebar.slider("Figure Width", 10, 30, 20)
            fig_height = st.sidebar.slider("Figure Height", 10, 30, 20)
            font_size = st.sidebar.slider("Font Size", 6, 16, 10)
            node_size = st.sidebar.slider("Node Size", 100, 500, 200)
            
            # Create and display the visualization
            st.header(f"Schema Mapping (Threshold: {threshold:.2f})")
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Draw the network
            node_colors = [G.nodes[node]["color"] for node in G.nodes]
            edge_colors = []
            
            for edge in G.edges():
                source_table = edge[0].split('(')[-1].strip(')')
                edge_colors.append(color_map[source_table])
            
            nx.draw(
                G, 
                pos,
                with_labels=True,
                node_color=node_colors,
                font_size=font_size,
                font_weight="bold",
                edge_color=edge_colors,
                node_size=node_size,
                ax=ax
            )
            
            plt.title(f"Schema Mapping Network (Threshold: {threshold:.2f})")
            st.pyplot(fig)
            
            # Display table connections
            st.header("Table Connections")
            connection_count = 0
            
            # Custom CSS for larger connection text
            st.markdown("""
            <style>
                .large-text {
                    font-size: 22px !important;
                    line-height: 1.5;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a single HTML string for all connections
            connections_html = ""
            for edge in G.edges():
                source = edge[0].split('\n')
                target = edge[1].split('\n')
                connections_html += f"<div class='large-text'>• {source[0]} ({source[2].strip('()')}) → {target[0]} ({target[2].strip(')')}) </div>"
                connection_count += 1
            
            # Display all connections with the larger text size
            if connection_count > 0:
                st.markdown(connections_html, unsafe_allow_html=True)
            else:
                st.markdown("<div class='large-text'>No connections found at this threshold. Try lowering the threshold value.</div>", unsafe_allow_html=True)
        else:
            st.warning("Please select at least two tables to generate the mapping visualization.")
    else:
        st.info("Please upload a file to begin.")

if __name__ == "__main__":
    main()
