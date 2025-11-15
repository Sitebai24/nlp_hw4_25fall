import os
import argparse
from tqdm import tqdm
import pickle
from torch.amp import GradScaler, autocast 

 

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--baseline', action='store_true',
                    help="Run pretrained T5 without training")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    # model_type = 'ft' if args.finetune else 'scr'
    if args.finetune:
        model_type = 'ft'
    elif args.baseline:
        model_type = 'bl'
    else:
        model_type = 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    #gt_record_path = os.path.join(f'records/dev_gt_records.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler, scaler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, optimizer, scheduler, best=True)
            print(f"✅ New best model saved (Dev F1={best_f1:.4f})")
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, optimizer, scheduler, best=False)
        # if epochs_since_improvement == 0:
        #     save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            print("⏸️ Early stopping triggered.")
            break
        #print("\n🔁 Loading the best checkpoint for test evaluation...")


def train_epoch(args, model, train_loader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # (推荐) 设置一个梯度裁剪的阈值
    max_grad_norm = 1.0 

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        # 1. 总是先清空梯度
        optimizer.zero_grad()
        
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # ✅ 2. 使用 'autocast' 来运行前向传播
        # 这会自动以 FP16 (混合精度) 运行
         # 确保引入
        with autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

        # ✅ 3. 使用 'scaler' 来缩放损失并反向传播
        scaler.scale(loss).backward()
        
        # ✅ 4. (推荐) 梯度裁剪
        # 在 scaler.step() 之前，先 unscale 梯度
        scaler.unscale_(optimizer)
        # 现在对“正常”的梯度进行裁剪，防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 

        # ✅ 5. Scaler 更新优化器
        scaler.step(optimizer)

        # ✅ 6. 更新 Scaler 的缩放因子
        scaler.update()

        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens

# original version of the train epoch 
# def train_epoch(args, model, train_loader, optimizer, scheduler):
#     model.train()
#     total_loss = 0
#     total_tokens = 0
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

#     for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
#         optimizer.zero_grad()
#         encoder_input = encoder_input.to(DEVICE)
#         encoder_mask = encoder_mask.to(DEVICE)
#         decoder_input = decoder_input.to(DEVICE)
#         decoder_targets = decoder_targets.to(DEVICE)

#         logits = model(
#             input_ids=encoder_input,
#             attention_mask=encoder_mask,
#             decoder_input_ids=decoder_input, 
#         )['logits']

#         non_pad = decoder_targets != PAD_IDX
#         loss = criterion(logits[non_pad], decoder_targets[non_pad])
#         loss.backward()
#         optimizer.step()
#         if scheduler is not None: 
#             scheduler.step()

#         with torch.no_grad():
#             num_tokens = torch.sum(non_pad).item()
#             total_loss += loss.item() * num_tokens
#             total_tokens += num_tokens

#     return total_loss / total_tokens

def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Get the BOS token ID
    bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Generate SQL queries - WITH DECODER START TOKEN!
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_start_token_id=bos_token_id,  # ← ADD THIS!
                max_length= 256,  # Increased
                num_beams=8,     # Use beam search
                early_stopping= True,
                length_penalty=0.9

            )
            
            # Decode the generated queries
            for gen_ids in generated_ids:
                query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_queries.append(query)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    # Save generated queries and compute records
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    # Compute metrics
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    # Compute error rate
    num_errors = sum(1 for msg in error_msgs if msg != "")
    error_rate = num_errors / len(error_msgs) if len(error_msgs) > 0 else 0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Get the BOS token ID
    bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader, desc="Testing"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries - WITH DECODER START TOKEN!
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_start_token_id=bos_token_id,  # ← ADD THIS!
                max_length=256,
                num_beams=8,
                early_stopping=True
                
            )
            
            # Decode the generated queries
            for gen_ids in generated_ids:
                query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_queries.append(query)
    
    # Save generated queries and compute records
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    # Load back the error messages to compute error rate
    with open(model_record_path, 'rb') as f:
        _, error_msgs = pickle.load(f)
    
    num_errors = sum(1 for msg in error_msgs if msg != "")
    error_rate = num_errors / len(error_msgs) if len(error_msgs) > 0 else 0
    
    print(f"Test set: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
    print(f"Generated {len(generated_queries)} test queries")
    print(f"Saved to {model_sql_path} and {model_record_path}")
        

   


def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)


    ## please just go to the checkpoint for 
    #training the model 
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # 2. 定义路径
    #model_type = 'ft' if args.finetune else 'scr'
    if args.finetune:
        model_type = 'ft'
    elif args.baseline:
        model_type = 'bl'
    else:
        model_type = 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir # 确保 'train' 函数能获取到
    
    # 这是我们希望恢复的 "新" checkpoint
    resume_checkpoint_path = os.path.join(checkpoint_dir, 'last_model.pt')
    
    # 这是你昨晚的 "旧" best_model (只含权重)
    warm_start_path = os.path.join(checkpoint_dir, 'best_model.pt')

    # 3. 尝试恢复或暖重启
    if os.path.exists(resume_checkpoint_path):
        # 场景 A: 完美恢复 (你今天的训练中断后，从这里继续)
        print(f"🧩 Resuming training from (full) checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print("✅ Model, Optimizer, and Scheduler states loaded.")
    
    elif os.path.exists(warm_start_path):
        # 场景 B: 暖重启 (你现在的情况)
        print(f"⚠️ 'last_model.pt' (full checkpoint) not found.")
        print(f"🔥 Performing WARM RESTART from (weights only) '{warm_start_path}'...")
        
        checkpoint = torch.load(warm_start_path, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
             # 兼容你最老的、只保存了 state_dict 的文件
            model.load_state_dict(checkpoint) 
            
        print("✅ Loaded model weights from old best_model.")
        print("🚨 Optimizer and Scheduler are NEW.")
        print("🚨  (e.g., --learning_rate 1e-5) ")

    else:
        # 场景 C: 从头开始
        print("🚀 No checkpoint found. Starting training from scratch...")

    # 4. 开始训练
    # 确保你已经修复了 train() 函数内部的 save_model 调用！
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # 5. 评估
    print("\n✅ Training finished. Evaluating best model on Dev and Test sets...")
    
    # 注意：这里的 load_model_from_checkpoint 只需要加载 best_model 的权重，
    # 它的实现 (t5_utils.py) 也是对的 (只加载 'model_state_dict')，所以评估部分不用改。
    if args.baseline:
        print("⚠️ Baseline mode: skipping checkpoint loading.")
    else:
        model = load_model_from_checkpoint(args, best=True)
    #model = load_model_from_checkpoint(args, best=True) 
    model.eval()


    # Dev set
    experiment_name = args.experiment_name
    #'ft_experiment'
    #model_type = 'ft' if args.finetune else 'scr'
    if args.finetune:
        model_type = 'ft'
    elif args.baseline:
        model_type = 'bl'
    else:
        model_type = 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print("Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
