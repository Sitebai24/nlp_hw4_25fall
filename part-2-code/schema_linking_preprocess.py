# hw4/hw4-code/part-2-code/schema_linking_preprocess.py
# hw4/hw4-code/part-2-code/schema_linking_preprocess.py

# hw4/hw4-code/part-2-code/schema_linking_preprocess.py
# hw4/hw4-code/part-2-code/schema_linking_preprocess.py
from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm
import os

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    tables = []
    for table_name, info in schema['ents'].items():
        cols = list(info.keys())[:5]  # 取前 5 个列名
        desc = f"Table {table_name}: " + ", ".join(cols)
        tables.append((table_name, desc))
    return tables

def retrieve_top_k_tables(question, tables, model, k=3):
    q_emb = model.encode(question, normalize_embeddings=True)
    table_texts = [t[1] for t in tables]
    table_embs = model.encode(table_texts, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, table_embs)[0]
    top_idx = scores.topk(k=min(k, len(scores))).indices.tolist()
    return [table_texts[i] for i in top_idx]

def generate_schema_linked_data(input_path, output_path, model, tables, k=3):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in tqdm(f_in, desc=f"Linking schema for {os.path.basename(input_path)}"):
            question = line.strip()
            if not question:
                continue

            linked_tables = retrieve_top_k_tables(question, tables, model, k)
            schema_text = " | ".join(linked_tables)  # 单行拼接
            linked_input = (
                f"Convert this natural language question into SQL using the following schema: "
                f"{schema_text} || Question: {question}"
            )
            linked_input = linked_input.replace("\n", " ").strip()
            f_out.write(linked_input + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    schema_path = os.path.join(data_dir, "flight_database.schema")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    tables = load_schema(schema_path)

    for split in ["train", "dev", "test"]:
        input_path = os.path.join(data_dir, f"{split}.nl")
        output_path = os.path.join(data_dir, f"{split}_with_schema.nl")
        print(f"🔗 Generating {output_path} ...")
        generate_schema_linked_data(input_path, output_path, model, tables, k=3)

    print("✅ All splits processed successfully.")
