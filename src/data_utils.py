from sklearn.model_selection import train_test_split
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast


class PrepareData:
    def __init__(self,file, subset=None):
        self.data_dir="data"
        # длина предложения
        seq_len = 2
        self.load(file)
        # "чистим" тексты
        self.cleaned_texts = list(map(self.clean_string, self.lines))
        if subset:
            self.cleaned_texts=self.cleaned_texts[:subset]

        # удаляем слишком короткие тексты
        self.cleaned_texts = [line for line in self.cleaned_texts if len(line.split()) >= seq_len]

        print("Все тексты:", len(self.cleaned_texts))
        print(self.cleaned_texts[:2])

        self.split_texts(0.8,0.5)
        print(f"Размеры: train:{len(self.train_texts)}, val: {len(self.val_texts)}, test: {len(self.test_texts)}")
        self.save_texts()

    # функция для "чистки" текстов
    def clean_string(self,text):
        # приведение к нижнему регистру
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)  # Удалить ссылки
        text = re.sub(r'[@#]\w+', '', text)  # Удалить @user #хэштеги
        # удаление всего, кроме латинских букв, цифр и пробелов
        text = re.sub(r'[^a-z0-9\s\']', '', text)
        # удаление дублирующихся пробелов, удаление пробелов по краям
        text = re.sub(r'\s+', ' ', text).strip()
        # return text if len(text) > 3 else ""
        return text

    def split_texts(self, train_ratio=0.8, val_ratio=0.5):
        self.train_texts, temp_texts = train_test_split(self.cleaned_texts, train_size=train_ratio, random_state=42, shuffle=False)
        
        # Потом val/test из остатка
        self.val_texts, self.test_texts = train_test_split(temp_texts, train_size=val_ratio, random_state=42, shuffle=False)

    def load(self,file):
        with open(file, 'r', encoding='utf-8') as f:
            self.lines = f.read().splitlines()

    def save_texts(self):
        with open(f"{self.data_dir}/train_texts.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.train_texts))
        with open(f"{self.data_dir}/val_texts.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.val_texts))
        with open(f"{self.data_dir}/test_texts.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.test_texts))


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer,max_len):
        self.samples = []
        self.max_len=max_len
        for line in texts:
            encoded = tokenizer(
                line,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded['input_ids'].squeeze(0)[:-1]  
            target_ids = encoded['input_ids'].squeeze(0)[1:] 
            self.samples.append((input_ids,target_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
        # return torch.tensor(x), torch.tensor(y)  # input, target


# encoded=tokenizer(
#                 f"[SOS] {cleaned_texts[0]} [EOS]",
#                 max_length=50,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt",
#                 add_special_tokens=False
#             )

# print(encoded)
# print(tokenizer.decode(encoded['input_ids'].squeeze(0),skip_special_tokens=True,clean_up_tokenization_spaces=True))

# sys.exit()


def prepare_tokenized(file):

    max_len=64
    with open("data/train_texts.txt", 'r', encoding='utf-8') as f:
        train_lines = f.read().splitlines()
    with open("data/val_texts.txt", 'r', encoding='utf-8') as f:
        val_lines = f.read().splitlines()
    with open("data/test_texts.txt", 'r', encoding='utf-8') as f:
        test_lines = f.read().splitlines()

    tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = TextDataset(train_lines, tokenizer,max_len)
    val_ds = TextDataset(val_lines, tokenizer,max_len)
    test_ds = TextDataset(test_lines, tokenizer,max_len)

    print(len(train_ds),len(val_ds),len(test_ds))
    # print(train_ds[:2],"\n",val_ds[:2],"\n",test_ds[:2],)

    # train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    # test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # val, targ=next(iter(train_loader))
    # print(tokenizer.decode(val[0],),"\n",tokenizer.decode(targ[0]))

    data_to_save= {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
        'max_len': max_len
    }

    torch.save(data_to_save,file)



if __name__ == "__main__":


    # data=PrepareData('data/tweets.txt',100000)

    # prepare_tokenized('data/texts_tokenized.pt')
    exit()



