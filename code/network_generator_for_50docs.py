import os
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import json_parser as jsp
import argparse
model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)

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

def create_embeddings(table_data, table_name, model,numerical=1):
    embeddings = [] 
    if numerical==1:
      for idx, item in enumerate(table_data):
          category, subcategory, data = item[0], item[1:-1], item[-1]
          combined_info_from_above=f'{item[0]} -> {item[1:-1]} -> {item[-1]}'
          emb_category = model.encode(category)
          emb_subcategory=0
          for index,value in enumerate(subcategory):
              emb_subcategory+=model.encode(value)*(len(subcategory)-(index))/len(subcategory)
          emb_data = model.encode(data)
        #   combined_emb = (emb_category*1.0 + emb_subcategory*0.66 + emb_data*0.33) / 3.0    # this was for taking the average of the vectors
          combined_emb = np.concatenate([emb_category,emb_subcategory,emb_data],axis=1)  # concatenate the vectors

          embeddings.append((table_name, category, '->'.join(subcategory), idx, combined_emb,combined_info_from_above))
    else:
      for idx, item in enumerate(table_data):
        category, subcategory, data = item[0], item[1:-1], item[-1]
        emb_category = model.encode(category)
        emb_subcategory=np.zeros_like(emb_category)
        for index,value in enumerate(subcategory):
          emb_subcategory+=model.encode(value)*(len(subcategory)-(index))/len(subcategory)
        # emb_subcategory = model.encode(subcategory)
        emb_data = model.encode(data)
        # print(emb_category.shape)

        # combined_emb = (emb_category + emb_subcategory + emb_data) / 3.0  # this was used for average task
        combined_emb=np.concatenate([emb_category,emb_subcategory,emb_data],axis=0)
        # print(combined_emb.shape)
        embeddings.append((table_name, category, subcategory, idx, combined_emb))
    return embeddings

def generate_color_palette(num_tables):
    base_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    if num_tables <= len(base_colors):
        return base_colors[:num_tables]
    else:
        return plt.cm.rainbow(np.linspace(0, 1, num_tables))

def create_similarity_network(information_to_be_checked, model,numerical,folder, file_name, similarity_threshold):
    # Process sentences into tables
    tables = process_sentences_to_tables(information_to_be_checked)

    # Generate embeddings for each table
    all_embeddings = []
    for table_name, table_data in tables.items():
        table_embeddings = create_embeddings(table_data, table_name, model, numerical)
        all_embeddings.extend(table_embeddings)

    # Prepare data for similarity computation
    # keys = [(t[0], t[1], t[3]) for t in all_embeddings]
    keys = [(t[0], t[1], t[3]) for t in all_embeddings]
    # data_to_be_mapped=[t[-1] for t in all_embeddings]
    embedding_matrix = [t[4] for t in all_embeddings]

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embedding_matrix)

    # Create graph
    G = nx.Graph()
    edges = set()

    # Generate color palette
    color_map = dict(zip(tables.keys(), generate_color_palette(len(tables))))

    # Add nodes with colors
    for t in all_embeddings:
        node = f"{t[1]}\n {t[2]}\n ({t[0]})"
        G.add_node(node, color=color_map[t[0]])

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i][0] != keys[j][0] and sim_matrix[i, j] >= similarity_threshold:
                node_i = f"{all_embeddings[i][1]}\n {all_embeddings[i][2]}\n ({keys[i][0]})"
                node_j = f"{all_embeddings[j][1]}\n {all_embeddings[j][2]}\n ({keys[j][0]})"
                edges.add((node_i, node_j))
    # print(edges)
    G.add_edges_from(edges)
    # for edge in G.edges():
    #     print('Here we are',edge)
    x_spacing = 2
    y_spacing = 2
    pos = {}

    for table_idx, table_name in enumerate(tables.keys()):
        table_nodes = [node for node in G.nodes() if table_name in node]
        for i, node in enumerate(table_nodes):
            pos[node] = (table_idx * x_spacing, i * y_spacing)
    plt.figure(figsize=(20, 20))
    node_colors = [G.nodes[node]["color"] for node in G.nodes]
    edge_colors = []
    # print('The data to be mapped',data_to_be_mapped)
    # to_be_reserved=[]
    # def reserve_mappings(edge):
    #     # to_be_reserved.append()
    #     alpha=[p.replace('\n','->') for p in edge]
    #     return to_be_reserved.append(alpha)
    
    for edge in G.edges():
        # print('We went to reserve mappings',
        # reserve_mappings(edge)
        # print('The edge here is',edge)
        source_table = edge[0].split('(')[-1].strip(')')
        edge_colors.append(color_map[source_table])
    # print('Here are the mappings that are to be resereved ',to_be_reserved)
    print("Hang tight while we are working with plotting images")
    nx.draw(G, pos,with_labels=True,node_color=node_colors,font_size=10,font_weight="bold",edge_color=edge_colors,node_size=200)
    plt.title("Sentence Similarity Network with Color-Coded Mappings")
    try:
        plt.savefig(f'/home/sa23k/embedding_based_schema_mapping/mappings/{folder}/concatenate_{file_name}/{similarity_threshold*100}_threshold.png')
    except:
        os.makedirs(f'/home/sa23k/embedding_based_schema_mapping/mappings/{folder}/concatenate_{file_name}',exist_ok=True)
        plt.savefig(f'/home/sa23k/embedding_based_schema_mapping/mappings/{folder}/concatenate_{file_name}/{similarity_threshold*100}_threshold.png')
    plt.show()
    return G, pos

# if __name__=='__main__':
#     parser=argparse.ArgumentParser(description="Process some arguments")
#     parser.add_argument('--json_file',required=True,help='path to the json file, please place your files  in json_files folder and just pass the name of the file')
#     parser.add_argument('--threshold',required=True,help='Cosine Threshold to filter the mappings')
#     parser.add_argument("--numerical",required=True,help="Boolean for numerical data, if your data is numerical pass 1, else pass 0")
#     args=parser.parse_args()
#     with open(f"/home/sa23k/embedding_based_schema_mapping/json_files/{args.json_file}.json","r") as f:
#         information=json.load(f)
#         G, pos = create_similarity_network(jsp.VMD_mapper(information), model,int(args.numerical),args.json_file,similarity_threshold=float(args.threshold))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some arguments")
    parser.add_argument('--json_file', required=True, help='path to the json file, please place your files in json_files folder and just pass the name of the file')
    parser.add_argument('--threshold', required=True, type=float, help='Cosine Threshold to filter the mappings')
    parser.add_argument("--numerical", required=True, type=int, choices=[0, 1], help="Boolean for numerical data, if your data is numerical pass 1, else pass 0")
    parser.add_argument("--table1", action="store_true", help="Include Table 1")
    parser.add_argument("--table2", action="store_true", help="Include Table 2")
    parser.add_argument("--table3", action="store_true", help="Include Table 3")
    args = parser.parse_args()

    with open(f"/home/sa23k/embedding_based_schema_mapping/json_files/{args.json_file}.json", "r") as f:
        information = json.load(f)
        
    # Filter the information based on table flags
    filtered_information = {}
    filename=''
    if args.table1:
        filtered_information["Table 1"] = information.get("Table 1", {})
        filename+='tab1'
    if args.table2:
        filtered_information["Table 2"] = information.get("Table 2", {})
        filename+='tab2'
    if args.table3:
        filtered_information["Table 3"] = information.get("Table 3", {})
        filename+='tab3'
    # print(filtered_information)
    G, pos = create_similarity_network(jsp.HMD_mapper(filtered_information), model, args.numerical, args.json_file, filename, similarity_threshold=args.threshold)


