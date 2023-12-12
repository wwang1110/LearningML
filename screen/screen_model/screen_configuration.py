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
    temperature = 1.0

    clip_model_name = "openai/clip-vit-base-patch32"
    clip_trainable = False
    clip_vit_dim = 768
    clip_txt_dim = 512

    roberta_model_name = "xlm-roberta-base"
    roberta_trainable = False
    roberta_txt_dim = 768

    projection_dim = 256

    dropout = 0.1
    
    '''
    debug = False
    #image_path = "C:/Moein/AI/Datasets/Flicker-8k/Images"
    #captions_path = "C:/Moein/AI/Datasets/Flicker-8k"
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048

    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    text_max_length = 77
    
    metadata_encoder_model = "distilbert-base-uncased"
    metadata_embedding = 768
    metadata_tokenizer = "distilbert-base-uncased"
    metadata_max_length = 512


    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    '''