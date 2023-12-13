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
    num_train_epochs = 1
    batch_size = 32
    output_dir = './checkpoint'
    evaluation_strategy = "epoch"
    
    #train lora args
    finetune_optim = "adamw_torch"
    finetune_epochs = 1
    finetune_batch_size = 32
    finetune_output_dir = './lora_checkpoint'
    finetune_evaluation_strategy = "epoch"