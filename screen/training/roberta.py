
from transformers import AutoTokenizer, XLMRobertaModel


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
pooled_embeddings = last_hidden_states.detach().mean(dim=1)