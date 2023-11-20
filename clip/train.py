from .AvgMeter import AvgMeter
from .CFG import CFG
from tqdm import tqdm
from .utils import get_lr

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    t_count = 0
    f_count = 0
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss, pred, target = model(batch)
        pred_cls = [batch['cls_id'][i] for i in pred]
        target_cls = [batch['cls_id'][i] for i in target]
        for i in range(len(pred_cls)):
            if pred_cls[i] == target_cls[i]:
                t_count += 1
            else:
                f_count += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter, t_count, f_count


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    t_count = 0
    f_count = 0
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss, pred, target = model(batch)
        pred_cls = [batch['cls_id'][i] for i in pred]
        target_cls = [batch['cls_id'][i] for i in target]
        for i in range(len(pred_cls)):
            if pred_cls[i] == target_cls[i]:
                t_count += 1
            else:
                f_count += 1

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter, t_count, f_count

