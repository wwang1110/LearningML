import os 
print(os.getcwd())

from clip.CIFAR100_dataloader import build_cifar100_loaders
from clip.Pokemon_Dataset import build_pokemon_loaders
from transformers import DistilBertTokenizer
import itertools
from clip.CFG import CFG
from clip import CLIPModel
import torch
from clip.train import train_epoch, valid_epoch

def main():

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader, valid_loader = build_cifar100_loaders(tokenizer)
    #train_loader, valid_loader = build_pokemon_loaders(tokenizer)

    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    train_accuracy = []
    valid_accuracy = []
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        
        model.train()
        train_loss, train_t_count, train_f_count = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        train_accuracy.append(train_t_count / (train_t_count + train_f_count))
        print(f"Train Loss: {train_loss.avg:.5f}, accuracy: {train_t_count / (train_t_count + train_f_count):.5f}")
        
        
        model.eval()
        with torch.no_grad():
            valid_loss, valid_t_count, valid_f_count = valid_epoch(model, valid_loader)
        valid_accuracy.append(valid_t_count / (valid_t_count + valid_f_count))
        print(f"Valid Loss: {valid_loss.avg:.5f}, accuracy: {valid_t_count / (valid_t_count + valid_f_count):.5f}")

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

if __name__ == "__main__":
    main()