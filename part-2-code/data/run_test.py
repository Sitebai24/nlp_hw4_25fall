# I've have the best model 
def main():
    #Get key arguments 
    args = get_args() 
    if args.use_wandb: 
    # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier 
    setup_wandb(args)


    