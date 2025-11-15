import pickle
from collections import Counter

# ========== Helper functions ==========

def load_pickle(pkl_path):
    """Load (records, error_msgs) from a pickle file."""
    with open(pkl_path, "rb") as f:
        records, error_msgs = pickle.load(f)
    return records, error_msgs


def load_sql(sql_path):
    """Load model-generated SQLs (one per line)."""
    with open(sql_path, "r") as f:
        sqls = [line.strip() for line in f.readlines()]
    return sqls


def categorize_error(msg):
    """Categorize SQL error message into general type."""
    msg_low = msg.lower().strip()
    if msg_low == "":
        return None
    elif "unrecognized token" in msg_low:
        return "unrecognized_token"
    elif "syntax error" in msg_low:
        return "syntax_error"
    elif "no such table" in msg_low:
        return "missing_table"
    elif "no such column" in msg_low:
        return "missing_column"
    elif "ambiguous" in msg_low:
        return "ambiguous_column"
    elif "misuse of aggregate" in msg_low:
        return "aggregate_misuse"
    elif "near" in msg_low:
        return "syntax_near"
    else:
        return "other"



def analyze_errors(error_msgs):
    """Count error types and store one example index per type."""
    counts = Counter()
    examples = {}
    for i, msg in enumerate(error_msgs):
        err_type = categorize_error(msg)
        if err_type:
            counts[err_type] += 1
            examples.setdefault(err_type, i)
    return counts, examples


def show_summary(name, counts, total):
    """Print summary table."""
    print(f"\n📊 --- {name} ---")
    total_errors = sum(counts.values())
    print(f"Total errors: {total_errors}/{total}  ({total_errors/total*100:.2f}%)")
    for err_type, count in counts.items():
        print(f"  {err_type:<20} {count:>4} ({count/total*100:.2f}%)")


def show_examples(name, sqls, error_msgs, examples):
    """Print example SQLs for each error type."""
    print(f"\n🔍 Example error SQLs for {name}:")
    for err_type, idx in examples.items():
        print(f"\n[{err_type.upper()}]  Example #{idx}")
        print(f"SQL:   {sqls[idx]}")
        print(f"Error: {error_msgs[idx]}")



# 修改成你想查看的 .pkl 文件路径
pkl_path = "records/t5_ft_ft_experiment_dev.pkl"  # fine-tuned
# pkl_path = "records/t5_bl_baseline_dev.pkl"     # baseline

# 1️⃣ 读取 pkl 文件
with open(pkl_path, "rb") as f:
    records, error_msgs = pickle.load(f)

print(f"Total {len(error_msgs)} samples loaded.")

# 2️⃣ 统计不同错误类型及数量
counter = Counter(error_msgs)
print(f"\n🧾 Found {len(counter)} unique error messages.\n")

# 3️⃣ 打印每种错误及数量
for err, count in counter.most_common():
    print(f"{count:>4} × {err}")


# ========== File paths (adjust if needed) ==========

baseline_sql_path = "results/t5_bl_baseline_dev.sql"
baseline_pkl_path = "records/t5_bl_baseline_dev.pkl"

finetune_sql_path = "results/t5_ft_ft_experiment_dev.sql"
finetune_pkl_path = "records/t5_ft_ft_experiment_dev.pkl"


# ========== Load baseline and fine-tuned results ==========

baseline_sqls = load_sql(baseline_sql_path)
_, baseline_errors = load_pickle(baseline_pkl_path)

finetune_sqls = load_sql(finetune_sql_path)
_, finetune_errors = load_pickle(finetune_pkl_path)

assert len(baseline_sqls) == len(baseline_errors), f"Length mismatch in baseline ({len(baseline_sqls)} vs {len(baseline_errors)})"
assert len(finetune_sqls) == len(finetune_errors), f"Length mismatch in finetuned ({len(finetune_sqls)} vs {len(finetune_errors)})"


# ========== Analyze both models ==========

baseline_counts, baseline_examples = analyze_errors(baseline_errors)
finetune_counts, finetune_examples = analyze_errors(finetune_errors)

# ========== Display summaries and examples ==========

show_summary("Baseline Model", baseline_counts, len(baseline_errors))
show_summary("Fine-tuned Model", finetune_counts, len(finetune_errors))

show_examples("Baseline Model", baseline_sqls, baseline_errors, baseline_examples)
show_examples("Fine-tuned Model", finetune_sqls, finetune_errors, finetune_examples)




# import pickle
# from collections import Counter

# # ========== 文件路径 ==========
# baseline_pkl_path = "records/t5_bl_baseline_dev.pkl"

# # ========== 载入 pickle ==========
# with open(baseline_pkl_path, "rb") as f:
#     records, error_msgs = pickle.load(f)

# print(f"✅ Loaded {len(error_msgs)} entries from {baseline_pkl_path}\n")

# # ========== 统计错误类型 ==========
# counter = Counter(error_msgs)
# unique_errors = len(counter)
# print(f"🧾 Found {unique_errors} unique error messages.\n")

# # ========== 打印错误统计 ==========
# for err, count in counter.most_common():
#     print(f"{count:>4} × {err}")

# # ========== 如果想看前几个非空错误示例 ==========
# print("\n🔍 First few actual SQL errors (non-empty only):")
# for i, e in enumerate(error_msgs):
#     if e.strip() != "":
#         print(f"\nExample #{i}")
#         print(f"Error: {e}")
#         print("-" * 80)
#     if i > 10:  # 限制输出前10个样本
#         break

