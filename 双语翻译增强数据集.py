# 下面是德语翻译成英语的例子，微改代码，可以实现英语翻译成德语，把所有的训练数据集都双向翻译。
import json
import torch
import random
import os
import pickle
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#设置随机种子
random.seed(42)
torch.cuda.empty_cache()
file_path='/kaggle/input/llms4subjetcs/GND-Subjects-all/GND-Subjects-all.json'
with open(file_path,'r') as f:
    data=json.loads(f.read())
gnd_id2name={}
all_codes=[]
for i in data:
    all_codes.append(i['Code'])
    gnd_id2name[i['Code']]=i['Name']
    for j in i['Alternate Name']:
        gnd_id2name[i['Code']]=gnd_id2name[i['Code']]+','+j

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
model=model.to("cuda")
# 示例德语句子
# german_sentence = "Dies ist ein Beispielsatz auf Deutsch."
# 对句子进行编码
def trans_de_en(german_sentences):
    with torch.no_grad():
        input_ids = tokenizer(german_sentences, return_tensors="pt", padding=True, truncation=True)
        # 使用模型生成翻译结果
        input_ids=input_ids.to('cuda')
        outputs = model.generate(**input_ids)
        # 对生成的结果进行解码
        translated_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        torch.cuda.empty_cache()
    return translated_sentences
def get_train_data(datas,x_train,isfanyi=False):
    cnt=0
    gnd_set=set()
    x_train1=[]
    for t in datas:
        if type(t['title'])!=str:
            t['title']=' '.join(t['title'])
        if type(t['abstract'])!=str:
            t['abstract']=' '.join(t['abstract'])
        # s1s.append(t['title']+'.'+t['abstract'])
        codes=[]
        for gnd in t['dcterms:subject']:
            if type(gnd)==dict:
                gnd_id=gnd['@id']
                codes.append(gnd_id)
            else:
                continue
        item1={}
        item1['sentence1']=t['title']+'.'+t['abstract']
        item1['label']=1
        item1['sentence2']='These are the names of the subjects:'
        item2={}
        item2['sentence1']=t['title']+'.'+t['abstract']
        item2['label']=0
        item2['sentence2']='These are the names of the subjects:'
        for gnd in t['dcterms:subject']:
            if type(gnd)==dict:
                gnd_id=gnd['@id']
            else:
                continue
            cnt+=1
            if gnd_id in gnd_id2name:
                gnd_set.add(gnd_id)
                gnd_val=gnd_id2name[gnd_id]
                item1['sentence2'] += gnd_val+','
                # 随机取出属于all_codes但不属于codes的gnd_id
                while True:
                    gnd_id=random.choice(all_codes)
                    if gnd_id not in codes:
                        break
                gnd_val=gnd_id2name[gnd_id]
                item2['sentence2'] += gnd_val+','
        x_train.append(item1)
        x_train1.append(item2)
    # 以batch_size批量把x_train中的句子翻译成英文
    batch_size=32
    if isfanyi:
        x_train_s1=[item['sentence1'] for item in x_train]
        x_train_s1_en=[]
        # 每batch_size个句子翻译一次,余下的句子也翻译一次
        i=0
        while i<len(x_train_s1):
            x_train_s1_en+=trans_de_en(x_train_s1[i:i+batch_size])
            i+=batch_size
            print(i)
        with open('x_train_de2en_s1.json', 'w', encoding='utf-8') as f:
            json.dump(x_train_s1_en, f, ensure_ascii=False)
        if len(x_train_s1_en)!=len(x_train):
            print('len(x_train_s1_en)!=len(x_train)')
        for i in range(len(x_train)):
            x_train[i]['sentence1']=x_train_s1_en[i]
            x_train1[i]['sentence1']=x_train_s1_en[i]
        
        x_train_s1=[item['sentence2'] for item in x_train]
        x_train_s1_en=[]
        # 每batch_size个句子翻译一次,余下的句子也翻译一次
        i=0
        while i<len(x_train_s1):
            x_train_s1_en+=trans_de_en(x_train_s1[i:i+batch_size])
            i+=batch_size
            print(i)
        with open('x_train_de2en_s2_1.json', 'w', encoding='utf-8') as f:
            json.dump(x_train_s1_en, f, ensure_ascii=False)
        if len(x_train_s1_en)!=len(x_train):
            print('len(x_train_s1_en)!=len(x_train)')
        for i in range(len(x_train)):
            x_train[i]['sentence2']=x_train_s1_en[i]
            
        x_train_s1=[item['sentence2'] for item in x_train1]
        x_train_s1_en=[]
        # 每batch_size个句子翻译一次,余下的句子也翻译一次
        i=0
        while i<len(x_train_s1):
            x_train_s1_en+=trans_de_en(x_train_s1[i:i+batch_size])
            i+=batch_size
            print(i)
        with open('x_train_de2en_s2_0.json', 'w', encoding='utf-8') as f:
            json.dump(x_train_s1_en, f, ensure_ascii=False)
        if len(x_train_s1_en)!=len(x_train1):
            print('len(x_train_s1_en)!=len(x_train)')
        for i in range(len(x_train1)):
            x_train1[i]['sentence2']=x_train_s1_en[i]
    x_train.extend(x_train1)

x_train=[]
datas=[]
with open('/kaggle/input/llms4subjetcs/all-subjects/all-subjects_dev_de.json','r') as f:
    datas+=json.loads(f.read())
get_train_data(datas,x_train,True)
# 将 x_train 和 x_dev 存储为 JSON 文件
with open('x_train_de2en.json', 'w', encoding='utf-8') as f:
    json.dump(x_train, f, ensure_ascii=False)
