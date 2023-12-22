from transformers import PretrainedConfig

clip_models = [
    #projection_dim = 512
    "openai/clip-vit-base-patch32", 
    #projection_dim = 768
    "openai/clip-vit-large-patch14"
    ]
xlm_roberta_models = [
    "xlm-roberta-base", 
    "xlm-roberta-large"
    ]

class ScreenConfiguration(PretrainedConfig):

    clip_model_name = "openai/clip-vit-base-patch32"
    clip_trainable = False
    clip_vit_dim = 768
    clip_txt_dim = 512

    roberta_model_name = "xlm-roberta-base"
    roberta_trainable = False
    roberta_txt_dim = 768

    projection_dim = 256

    dropout = 0.1
        
    #train args
    optim = "adamw_torch"
    num_train_epochs = 10
    batch_size = 32

    save_strategy = "steps"
    output_dir = './checkpoint'
    save_steps = 10
    save_total_limit = 5
    
    evaluation_strategy = "steps"
    eval_steps = 5

    logging_strategy = "steps"
    logging_dir = './logs'
    logging_steps = 5
    #report_to = 'azure_ml'
    report_to = 'tensorboard'

    learning_rate = 1e-4 # 1e-5
    dataloader_num_workers = 4
    gradient_accumulation_steps = 1
    
    #train lora args
    finetune_optim = "adamw_torch"
    finetune_epochs = 1
    finetune_batch_size = 32
    finetune_output_dir = './lora_checkpoint'
    finetune_evaluation_strategy = "epoch"