from transformers import PretrainedConfig

class MLPConfig(PretrainedConfig):
    model_type = "screen_mlp"

    def __init__(
        self,
        num_features: int = 256,
        num_classes: int = 3,
        **kwargs,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        super().__init__(**kwargs)