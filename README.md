# EmbMap---Embedding_Based_Schema_Mapper

To run this app locally without UI 
 cd code
 python network_generator_for_50docs.py --json_file=Doc_52 --threshold=0.85 --numerical=0 --table1 --table2
 
Since Doc_52 in json_files is textual table flag --numerical=0
If the table is a numerical table, flag --numerical=1
The last 2 flags let the code know to use specific tables in json (Doc_52.json)

In the UI, you have the option to choose the style of embedding creation. 
1. Weighted Summation
2. Concatenation

Uses **HuggingFace sentence Transformers** library and model **'Lajavaness/bilingual-embedding-large'**
At this time, on the MTEB dashboard, **Lajavanesss** performance exceeds the openAI embedding model (text-embedding-3-s,l) and other models whose weights are far bigger than the bilingual-embedding-large

_Weighted Sum Embeddings_
And once we have the vectors for each level (header, subheader, its subheader, etc, data)
We add all the vectors to create 1 final vector for the given metadata row or column. 

Tables with numerical data: Since the numerical data often comes without context and units, numbers without context and units often mislead the embedding vector. Hence, we utilized variable weight assignments to the tables that contain numerical data to mitigate the misleading caused by numerical data.

Weight to current level in the table = (Total no of levels - current level + 1) / (Total no of levels) 

So, for Root header = (3-1+1)/3 = 1

Sub root = (3-2+1)/3=2/3=0.6

Data = (3-3+1)/3 = 0.33

We place high emphasis on root attributes, and weights from sub roots to Data are based on their hierarchy level in the table. Such that Data(numerical) has the least influence on the final embedding.


_Non-Weight Embeddings_
Tables with textual data: Uses constant weights for the textual metadata and data since the textual data is beneficial for creating embedding vectors. Hence, we assigned the same weightage irrespective of its position in the table.

**Weighted Summation **
Once we have these weighted vectors or non-weighted vectors for metadata and data, we add them to create 1 final vector.

**Concatenation**
Instead of creating 1 final vector by summing vectors (weighted or non-weighted) along the vector axis for all levels of metadata and data, we concatenated them into 1 long vector.


We calculate cosine similarity between the tables based on the user's approach (concatenation/summation) to perform schema matching. 


![image](https://github.com/user-attachments/assets/76efba67-2fce-418c-a87f-6eae406dd70a)




![image](https://github.com/user-attachments/assets/8935c244-ad23-4b83-b7d0-ab49497aecd3)

