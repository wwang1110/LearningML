def model_size_in_MB(model):
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def print_state_dict(model):
    for k, v in model.state_dict().items():
        print(
            f"count: {v.numel()} \t|| shape: {v.shape} \t|| cuda: {v.is_cuda} \t|| name: {k}"
        )

def print_model_info(model):
    model_size_in_MB(model)
    print_trainable_parameters(model)
    print_state_dict(model)