from load_data import T5Dataset
from transformers import T5TokenizerFast

dataset = T5Dataset('data', 'train')
tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

# Check first 3 examples
for i in range(3):
    item = dataset[i]
    nl = tokenizer.decode(item['encoder_input_ids'], skip_special_tokens=True)
    sql = tokenizer.decode(item['decoder_targets'], skip_special_tokens=True)
    print(f"\n--- Example {i} ---")
    print(f"Input (NL):  {nl}")
    print(f"Target (SQL): {sql}")