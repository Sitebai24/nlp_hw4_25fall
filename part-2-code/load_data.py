import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
import json
import sqlparse

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0 

def custom_clean_sql(sql_raw: str) -> str:
    """
    Specially customized SQL cleaner for this project's data.
    1. (If sqlparse available) Auto-formats keywords, spacing.
    2. (Custom) Removes "AND 1 = 1" ghost conditions.
    3. (Custom) Fixes " , " (space-comma) formatting.
    4. (Custom) Ensures space between keywords and parentheses (e.g., "AND(" -> "AND (").
    """
    sql = sql_raw

    # 1. (Recommended) Use sqlparse for general formatting
    if sqlparse is not None:
        try:
            sql = sqlparse.format(sql, keyword_case='upper', strip_comments=True)
        except Exception:
            pass  # If sqlparse fails, just use the raw string

    # 2. (Custom) Remove "1 = 1" ghost conditions
    sql = re.sub(r'\s+AND\s+1\s*=\s*1\b', ' ', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s+WHERE\s+1\s*=\s*1\s+AND\b', ' WHERE ', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s+WHERE\s+1\s*=\s*1\b', ' ', sql, flags=re.IGNORECASE)

    # 3. (Custom) Fix space-comma formatting
    sql = re.sub(r'\s+,\s+', ', ', sql)
    
    # 4. (Custom - NEW) Fix keyword-parenthesis spacing
    sql = re.sub(r'\bAND\(\s*', 'AND (', sql, flags=re.IGNORECASE)  # Fixes "AND("
    sql = re.sub(r'\bOR\(\s*', 'OR (', sql, flags=re.IGNORECASE)    # Fixes "OR("
    sql = re.sub(r'=\(\s*', '= (', sql)                            # Fixes "=("

    # 5. Final whitespace cleanup
    sql = ' '.join(sql.split())

    return sql

def summarize_schema(schema_path, max_tables=None, max_cols=6):
    """
    Parse a JSON schema file and build a concise text summary for T5 input.
    """
    with open(schema_path) as f:
        data = json.load(f)
    ents = data.get("ents", {})
    lines = []
    for i, (table, cols) in enumerate(ents.items()):
        if max_tables and i >= max_tables:
            break
        colnames = [v["utt"] for v in list(cols.values())[:max_cols]]
        lines.append(f"{table}({', '.join(colnames)})")
    return "\n".join(lines)

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.data_folder = data_folder
        
        # Initialize T5 tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        # Define max sequence length (you can adjust this if needed)
        self.max_length = 640 
        
        # Use extra_id_0 as the BOS token for decoder
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')

                # Load and summarize schema once
        # schema_path = os.path.join(data_folder, "flight_database.schema")
        # if os.path.exists(schema_path):
        #     print(f"✅ Using schema from {schema_path}")
        #     self.schema_summary = summarize_schema(schema_path, max_tables=15, max_cols=6)
        # else:
        #     print("⚠️ Schema file not found — using no schema context.")
        #     self.schema_summary = ""
        # Skip schema summarization (already handled by schema linking)
        # Skip schema summarization (already handled by schema linking)
        print("⚠️ Skipping schema summarization since schema linking handles it.")
        self.schema_summary = ""
        
        # Process data
        self.data = self.process_data(data_folder, split, self.tokenizer)


    def process_data(self, data_folder, split, tokenizer):

        nl_path_with_schema = os.path.join(data_folder, f'{split}_with_schema.nl')

        
        if os.path.exists(nl_path_with_schema):
            print(f"✅ Found schema-linked data: {nl_path_with_schema}")
            nl_path = nl_path_with_schema
            use_schema_linking = True
        else:
            nl_path = os.path.join(data_folder, f'{split}.nl')
            print(f"⚠️ Using plain NL data (no schema linking): {nl_path}")
            use_schema_linking = False
        # Load natural language queries
        # nl_path = os.path.join(data_folder, f'{split}.nl')

        with open(nl_path, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        processed_data = []
        
        # For test set, we don't have SQL queries
        if split == 'test':
            for nl in nl_queries:
                # Tokenize input (natural language)
                #nl = "Convert this natural language question into SQL: " + nl
                # if use_schema_linking:
                #     prompt = nl
                # else:
                #     prompt = (
                #     f"You are an expert SQL generator. Using the following schema:\n"
                #      f"{schema_text}\n\n"
                #      f"Generate a valid SQL query that can run successfully.\n"
                #     f"Question: {nl}\nSQL:"
                #         )
                if use_schema_linking:
                    # prompt = "Convert this natural language question into SQL: " + nl
                    prompt = nl
                else:
                    prompt = (
                        f"You are an expert SQL generator. Using the following schema:\n"
                        f"{self.schema_summary}\n\n"
                        f"Generate a valid SQL query that can run successfully.\n"
                        f"Question: {nl}\nSQL:"
                    )
                #     prompt = (
                #     f"Given the following database schema:\n{self.schema_summary}\n\n"
                #     f"Translate the following question into a SQL query.\n"
                #     f"Question: {nl}\nAnswer:"
                # )   
                # prompt = (
                #     f"Given the following database schema:\n{self.schema_summary}\n\n"
                #     f"Translate the following question into a SQL query.\n"
                #     f"Question: {nl}\nAnswer:"
                # )
                encoder_input = tokenizer(
                    prompt,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                
                processed_data.append({
                    'encoder_input_ids': encoder_input['input_ids'].squeeze(0),
                    'encoder_attention_mask': encoder_input['attention_mask'].squeeze(0),
                    'decoder_input_ids': None,  # No target for test
                    'decoder_targets': None
                })
        else:
            # Load SQL queries for train/dev
            sql_path = os.path.join(data_folder, f'{split}.sql')
            with open(sql_path, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]
            
            
            assert len(nl_queries) == len(sql_queries), \
                f"Mismatch between NL and SQL queries: {len(nl_queries)} vs {len(sql_queries)}"
            
            for nl, sql_raw in zip(nl_queries, sql_queries):
                # Tokenize input (natural language)
                if use_schema_linking:
                    # prompt = "Convert this natural language question into SQL: " + nl
                    prompt = nl
                else:
                    prompt = (
                                f"You are an expert SQL generator. Using the following schema:\n"
                                f"{self.schema_summary}\n\n"
                                f"Generate a valid SQL query that can run successfully.\n"
                                f"Question: {nl}\nSQL:"
                            )
                # prompt = (
                #     f"Given the following database schema:\n{self.schema_summary}\n\n"
                #     f"Translate the following question into a SQL query.\n"
                #     f"Question: {nl}\nAnswer:"
                # )
                encoder_input = tokenizer(
                    prompt,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )

                sql_clean = custom_clean_sql(sql_raw)
                
                # Tokenize output (SQL query)
                decoder_output = tokenizer(
                    sql_clean,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                
                processed_data.append({
                    'encoder_input_ids': encoder_input['input_ids'].squeeze(0),
                    'encoder_attention_mask': encoder_input['attention_mask'].squeeze(0),
                    'decoder_input_ids': decoder_output['input_ids'].squeeze(0),
                    'decoder_targets': decoder_output['input_ids'].squeeze(0)
                })
        
        return processed_data
    
    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        return self.data[idx] 
     

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    
    # Extract components from batch
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    encoder_attention_mask = [item['encoder_attention_mask'] for item in batch]
    decoder_input_ids_list = [item['decoder_input_ids'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
    
    # For decoder, we need to:
    # 1. Prepend BOS token (<extra_id_0>) to create decoder inputs
    # 2. Use the original tokens as targets (what the decoder should predict)
    
    # Get BOS token ID (using the tokenizer to get extra_id_0)
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    
    decoder_inputs_list = []
    decoder_targets_list = []
    initial_decoder_inputs_list = []
    
    for decoder_ids in decoder_input_ids_list:
        # Decoder input: <BOS> + tokens[:-1]
        # This shifts the sequence right by prepending BOS
        decoder_input = torch.cat([torch.tensor([bos_token_id]), decoder_ids[:-1]])
        decoder_inputs_list.append(decoder_input)
        
        # Decoder target: original tokens (what we want to predict)
        decoder_targets_list.append(decoder_ids)
        
        # Initial decoder input for generation (just the BOS token)
        initial_decoder_inputs_list.append(torch.tensor([bos_token_id]))
    
    # Pad decoder inputs and targets
    decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
    
    # For initial decoder inputs, we just need the BOS token for each example in the batch
    initial_decoder_inputs = torch.tensor([[bos_token_id] for _ in batch])
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Extract components from batch
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    encoder_attention_mask = [item['encoder_attention_mask'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
    
    # Get BOS token ID for initial decoder input
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    
    # Initial decoder input for generation (just the BOS token for each example)
    initial_decoder_inputs = torch.tensor([[bos_token_id] for _ in batch])
    
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")


    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Load training data
    train_nl_path = os.path.join(data_folder, 'train.nl')
    train_sql_path = os.path.join(data_folder, 'train.sql')
    train_x = load_lines(train_nl_path)
    train_y = load_lines(train_sql_path)
    
    # Load dev data
    dev_nl_path = os.path.join(data_folder, 'dev.nl')
    dev_sql_path = os.path.join(data_folder, 'dev.sql')
    dev_x = load_lines(dev_nl_path)
    dev_y = load_lines(dev_sql_path)
    
    # Load test data (no labels)
    test_nl_path = os.path.join(data_folder, 'test.nl')
    test_x = load_lines(test_nl_path)
    
    return train_x, train_y, dev_x, dev_y, test_x