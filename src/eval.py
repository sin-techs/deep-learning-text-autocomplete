from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast
# from data_utils import TextDataset
import torch
import torch.nn.functional as F
# from lstm_model import MyRNN
import evaluate
# from eval_transformer_pipeline import generate_autocomplete_gpt2

# tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
# tokenizer.pad_token = tokenizer.eos_token

# data=torch.load('data/texts_tokenized.pt',weights_only=False)
# print("Loaded: ",len(data['train_ds']),len(data['val_ds']),len(data['test_ds']))
# # train_loader = DataLoader(data['train_ds'], batch_size=128, shuffle=True)
# # val_loader = DataLoader(data['val_ds'], batch_size=64, shuffle=False)
# test_loader = DataLoader(data['test_ds'], batch_size=64, shuffle=False)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vocab_size=tokenizer.vocab_size
# model = MyRNN(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256,padding_idx=tokenizer.pad_token_id).to(device)
# model.load_state_dict(torch.load('models/model_weights.pth'))
# model.eval()
# criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

def calc_rouge(pred, ref):
    rouge = evaluate.load("rouge")
    res = rouge.compute(predictions=pred, references=ref)
    return res['rouge1'], res['rouge2']

def generate_autocomplete(model, tokenizer, prompt, max_new_tokens=5, 
                         temperature=0.8, top_k=50, device='cuda',max_length=63):
    model.eval()
    
    # Токенизируем промпт
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # print(f"Промпт: '{prompt}' → {input_ids.shape[1]} токенов")
    
    with torch.no_grad():
        generated = input_ids
        for step in range(max_new_tokens):
            context = generated[:, -max_length:]
            
            # Предсказываем следующий токен
            with torch.no_grad():
                logits = model(context)[:, -1, :]  # [1, vocab_size]
                token_ids = torch.argmax(logits, dim=-1)
                # print(tokenizer.decode(token_ids, skip_special_tokens=True))
            # Top-K + Temperature sampling
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Добавляем токен
            generated = torch.cat([generated, next_token], dim=1)
            
            # Декодируем для отладки (опционально)
            current_text = tokenizer.decode(generated[0])
            # if step % 5 == 0:
            # print(f"Шаг {step}: {current_text}")
    
    # Финальный результат
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return full_text



# print(data['test_ds'][0])

def eval_on_test(model,loader,autocomplete,tokenizer,ratio=0.75,pad_token_id=50256,demo=-1,device="cuda"):
    model.eval()
    r1_sum,r2_sum=0,0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            batch_size = x_batch.size(0)
            predictions = []
            references = []
            
            for i in range(batch_size):
                
                line=tokenizer.decode(x_batch[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                real_len=len(line.split())

                context_len = int(real_len * ratio)  # 75% для контекста
                target_len = real_len - context_len           # 25% для предсказания

                if (context_len>real_len):
                    continue
                # print(real_len,context_len,target_len)
                # print(x_batch[i])
                
                prompt=' '.join(line.split()[:context_len])
                words=autocomplete(model,tokenizer,prompt,max_new_tokens=target_len,temperature=0.5)
                if demo>0:
                    print()
                    print("prompt: ",prompt)
                    print("test:   ", line)
                    print("gen:    ", words)
                    demo-=1
                elif demo==0:
                    return 0,0
                predictions.append(' '.join(words.split()[context_len:]))
                references.append(' '.join(line.split()[context_len:]))
            r1,r2=calc_rouge(predictions,references)
            # print(r1,r2)
            r1_sum+=r1
            r2_sum+=r2
    return r1_sum/len(loader), r2_sum/len(loader)

# print(eval_on_test(model,test_loader,autocomplete=generate_autocomplete_gpt2,demo=-1))


# words=generate_autocomplete(model,tokenizer,"i hate car",max_new_tokens=20,temperature=0.5)
# print(len(words.split()),words)