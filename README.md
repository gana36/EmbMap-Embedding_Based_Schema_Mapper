# EmbMap - Embedding-Based Schema Mapper

An intelligent schema mapping tool that uses advanced embedding techniques to match table schemas based on semantic similarity.

## Overview

EmbMap leverages **HuggingFace Sentence Transformers** with the **'Lajavaness/bilingual-embedding-large'** model to perform schema matching between tables. This model currently outperforms OpenAI's embedding models (text-embedding-3-s, text-embedding-3-l) and other larger models on the MTEB dashboard.

## Features

- **Dual Embedding Strategies**: Choose between Weighted Summation and Concatenation approaches
- **Context-Aware Processing**: Different handling for textual vs numerical data
- **Hierarchical Weight Assignment**: Smart weighting based on table structure levels
- **Cosine Similarity Matching**: Accurate schema matching using semantic similarity

## Installation

```bash
cd code
```

## Usage

### Command Line Interface

```bash
python network_generator_for_50docs.py --json_file=Doc_52 --threshold=0.85 --numerical=0 --table1 --table2
```

#### Parameters

- `--json_file`: Specify the JSON file name (e.g., Doc_52 for Doc_52.json in json_files directory)
- `--threshold`: Similarity threshold for matching (e.g., 0.85)
- `--numerical`: Set to `1` for numerical tables, `0` for textual tables
- `--table1` and `--table2`: Flags to specify which tables to use from the JSON file

### User Interface

The UI provides options to choose between two embedding creation styles:
1. **Weighted Summation**
2. **Concatenation**

## Methodology

### Embedding Model

- **Model**: `Lajavaness/bilingual-embedding-large`
- **Library**: HuggingFace Sentence Transformers
- **Performance**: Superior to OpenAI embedding models on MTEB dashboard

### Embedding Strategies

#### 1. Weighted Summation

**For Numerical Tables:**
- Applies variable weight assignments to mitigate misleading effects of context-less numerical data
- Weight formula: `(Total levels - Current level + 1) / Total levels`

Example for a 3-level table:
- Root header: `(3-1+1)/3 = 1.0`
- Sub root: `(3-2+1)/3 = 0.67`
- Data: `(3-3+1)/3 = 0.33`

This hierarchy ensures numerical data has minimal influence while emphasizing structural metadata.

**For Textual Tables:**
- Uses constant weights across all levels
- Equal importance given to all textual metadata and data
- Beneficial since textual data provides meaningful context for embeddings

#### 2. Concatenation

Instead of summing weighted/non-weighted vectors, this approach concatenates all level vectors into one comprehensive vector, preserving all dimensional information.

### Schema Matching Process

1. **Vector Generation**: Create embeddings for each table level (headers, subheaders, data)
2. **Weight Application**: Apply appropriate weighting strategy based on data type
3. **Vector Combination**: Use either summation or concatenation approach
4. **Similarity Calculation**: Compute cosine similarity between final table vectors
5. **Schema Matching**: Match schemas based on similarity scores above threshold

## File Structure (To run it Local) .

```
├── code/
│   ├── network_generator_for_50docs.py
│── json_files/
│   └── Doc_52.json
│        
```

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.



![image](https://github.com/user-attachments/assets/76efba67-2fce-418c-a87f-6eae406dd70a)




![image](https://github.com/user-attachments/assets/8935c244-ad23-4b83-b7d0-ab49497aecd3)

