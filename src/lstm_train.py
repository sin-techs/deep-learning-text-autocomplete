import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# from src.data_utils import TextDataset #, text_collate_fn,prepare_dataset
# from src.lstm_model import MyRNN
import evaluate
rouge = evaluate.load("rouge")

# # with open('data/tweets_tokenized.pkl', 'rb') as f:
# #     data = pickle.load(f)



# # train_ds, val_ds, test_ds= prepare_dataset()
# # print("Loaded: ",len(train_ds),len(val_ds),len(test_ds))

# tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
# tokenizer.pad_token = tokenizer.eos_token

# # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, 
# #                           collate_fn=lambda batch: text_collate_fn(batch, tokenizer))
# # val_loader = DataLoader(train_ds, batch_size=64, shuffle=False, 
# #                         collate_fn=lambda batch: text_collate_fn(batch, tokenizer))
# # test_loader = DataLoader(train_ds, batch_size=64, shuffle=False,
# #                          collate_fn=lambda batch: text_collate_fn(batch, tokenizer))

# data=torch.load('data/texts_tokenized.pt',weights_only=False)
# print("Loaded: ",len(data['train_ds']),len(data['val_ds']),len(data['test_ds']))
# train_loader = DataLoader(data['train_ds'], batch_size=128, shuffle=True)
# val_loader = DataLoader(data['val_ds'], batch_size=128, shuffle=False)
# test_loader = DataLoader(data['test_ds'], batch_size=64, shuffle=False)

# # val, targ=next(iter(train_loader))
# # print(tokenizer.decode(val[0],skip_special_tokens=True,clean_up_tokenization_spaces=True))
# # print(tokenizer.decode(targ[0],skip_special_tokens=True,clean_up_tokenization_spaces=True))
# # exit()

# # создание модели
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device="cpu"
# print(device)
# vocab_size=tokenizer.vocab_size
# pad_token_id=tokenizer.pad_token_id
# model = MyRNN(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256,padding_idx=pad_token_id).to(device)
# print("vocab",vocab_size)
# print("pad token",pad_token_id)

# # создание оптимизатора и функции потерь
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # создайте оптимайзер с регуляризацией
# criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)


# код обучения одной эпохи
def train_epoch(model, loader,optimizer,criterion,device,vocab_size):
    model.train()
    total_loss = 0
    for x,y in tqdm(loader):
        x,y = x.to(device),y.to(device)
        # print("b ",x.shape, x.shape)
        x = x.reshape(-1, x.size(-1)) 

        optimizer.zero_grad() # обнулите градиенты

        logits = model(x) 
        logits = logits.reshape(-1, vocab_size)
        targets = y.reshape(-1) 
        # print("a ",logits.shape, targets.shape)

        loss = criterion(logits, targets) # посчитайте функцию потерь
        loss.backward() # посчитайте градиенты

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # примените gradient clipping
        optimizer.step() # обновите градиенты
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_train(model, loader,tokenizer,criterion,device,vocab_size):
    model.eval()
    sum_loss, total = 0, 0

    rouge1 = 0
    rouge2 = 0
    predictions = []
    references = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader):
            x_batch=x_batch.to(device)
            y_batch=y_batch.to(device)
            predictions = []
            references = []
            # проходим по батчу
            x_output = model(x_batch) # выход модели для входа x_batch
            for i in range(x_batch.size(0)):
                # x_context = x_batch[i]
                logits = x_output[i]
                token_ids = torch.argmax(logits, dim=-1)
                # Истинная цель
                ref_tokens = y_batch[i]
                
                pred_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                # print("pred",pred_text)
                # print("ref",ref_text)
                predictions.append(pred_text)
                references.append(ref_text)
            res=rouge.compute(predictions=predictions, references=references)
            rouge1+=res['rouge1']
            rouge2+=res['rouge2']

            x_output=x_output.reshape(-1, vocab_size)
            y_batch=y_batch.reshape(-1)
            loss = criterion(x_output, y_batch) # функция потерь
            sum_loss += loss.item() # суммарная функция потерь
    
    # лосс и accuracy
    avg_loss = sum_loss / len(loader)
    avg_rouge1 = rouge1 / len(loader)
    avg_rouge2 = rouge2 / len(loader)
    return avg_loss, avg_rouge1, avg_rouge2

def plot_data(train_losses, val_losses, r1s, r2s):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)


    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs,train_losses,'b-o', label='Train Loss', linewidth=2)
    plt.plot(epochs,val_losses, 'r-o', label='Val Loss', linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Loss по эпохам')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rouge
    plt.subplot(1, 2, 2)
    plt.plot(epochs,r1s, 'g-^',label='Rouge-1', linewidth=2)
    plt.plot(epochs, r2s, 'm-v', label='Rouge-2', linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Rouge')
    plt.title('Rouge по эпохам')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# # обучение
# train_losses, val_losses = [], []
# r1s, r2s = [], []

# for epoch in range(10):
#     loss = train_epoch(model, train_loader)
#     val_loss, r1, r2 = evaluate_train(model, val_loader)
#     train_losses.append(loss)
#     val_losses.append(val_loss)
#     r1s.append(r1)
#     r2s.append(r2)
#     print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Val_loss={val_loss}, Rouge1 = {r1:.4f}, Rouge2 = {r2:.4f}")

# plot_data(train_losses,val_losses,r1s,r2s)


# torch.save(model.state_dict(), 'models/model_weights.pth')