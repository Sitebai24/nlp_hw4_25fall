import os

#good ini
# init. xavier_uniform_(model.weights)
# xavier 
# data augenmentation 

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
        wandb.init(
        project="text-to-sql-t5",
        name=args.experiment_name,
        config=vars(args)
    )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        # Load pretrained T5-small model for finetuning
        print("Loading pretrained T5-small model for finetuning...")
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        model.config.dropout_rate = 0.05
        model.config.attention_dropout_rate = 0.05
    elif args.baseline:
        print("Loading pretrained T5-small for BASELINE (no training)...")
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        # Initialize T5-small from scratch (random weights)
        print("Initializing T5-small model from scratch...")
        config = T5Config.from_pretrained('google-t5/t5-small',  dropout_rate=0.1,
            attention_dropout_rate=0.1)
        model = T5ForConditionalGeneration(config)
    
    # num_layers_to_freeze = 1
    # for name, param in model.named_parameters():
    #     if name.startswith("encoder.block.") and int(name.split(".")[2]) < num_layers_to_freeze:
    #         param.requires_grad = False

    # 1️⃣ 冻结共享词嵌入层 (Embedding Layer)
    if hasattr(model, 'shared'):
        print("--- FREEZE: Shared Embedding Layer ---")
        for param in model.shared.parameters():
            param.requires_grad = False

    # 2️⃣ 冻结前 2 层 Encoder
    # it used to be 2, and convert it to 1 
    num_encoder_layers_to_freeze = 0
    print(f"--- FREEZE: First {num_encoder_layers_to_freeze} Encoder blocks ---")
    for i in range(num_encoder_layers_to_freeze):
        for param in model.encoder.block[i].parameters():
            param.requires_grad = False

    # 3️⃣ 冻结前 2 层 Decoder
    # it used to be 2, and convert it to 1 
    num_decoder_layers_to_freeze = 0
    print(f"--- FREEZE: First {num_decoder_layers_to_freeze} Decoder blocks ---")
    for i in range(num_decoder_layers_to_freeze):
        for param in model.decoder.block[i].parameters():
            param.requires_grad = False

    
    
    # ✅ Print how many layers are frozen
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"🔒 Frozen parameters: {frozen:,} / {total:,} ({frozen/total:.2%})")
    
    model = model.to(DEVICE)
    print(f"Model moved to {DEVICE}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, optimizer, scheduler, best):
    mkdir(checkpoint_dir)
    # Save model checkpoint to be able to load the model later
    if best:
        save_path = os.path.join(checkpoint_dir, 'best_model.pt')
        print(f"Saving best model to {save_path}")
        # added for the resuming training 
        torch.save({
            'model_state_dict': model.state_dict(),
        }, save_path)
    else:
        save_path = os.path.join(checkpoint_dir, 'last_model.pt')
        print(f"Saving last model to {save_path}")
         # added for the resuming training 
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }, save_path)
    

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    if best:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        print(f"Loading best model from {checkpoint_path}")
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'last_model.pt')
        print(f"Loading last model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Initialize model architecture
    model = initialize_model(args)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    return model



def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

