import os
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
import evaluate
import torch
import random
import multiprocessing

# --- Фиксируем сиды для воспроизводимости ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():    
    torch.manual_seed(SEED)

# --- Загружаем токенизатор и датасет ---
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
ds = load_dataset('amazon_polarity')
ds_train_all = ds['train']
ds_test_all = ds['test']

# --- Эвристическая классификация по типу отзыва ---
# жалоба 
COMPLAINT_KW = ['not', 'no', 'dirty', 'broken', 'complain', 'problem', 'issue', 'refund', 'late', 'rude', 'terrible']
# комплименты
PRAISE_KW = ['great', 'excellent', 'clean', 'friendly', 'love', 'amazing', 'perfect', 'good', 'recommend']
# вопрос
QUESTION_KW = ['how', 'what', 'where', 'when', 'why', 'is there', 'do you', 'could you']
# предложение 
SUGGESTION_KW = ['should', 'could', 'would be', 'suggest', 'recommendation', 'idea']

def classify_type_rule(text: str) -> str:
    t = text.lower()
    for kw in COMPLAINT_KW:
        if kw in t:
            return 'complaint'
    for kw in PRAISE_KW:
        if kw in t:
            return 'praise'
    for kw in QUESTION_KW:
        if kw in t:
            return 'question'
    for kw in SUGGESTION_KW:
        if kw in t:
            return 'suggestion'
    return 'other'

# --- Подготовка датафреймов ---
df_train_all = pd.DataFrame({
    'text': [x['content'] for x in ds_train_all],
    'labels': [x['label'] for x in ds_train_all]
}) 

df_test_all = pd.DataFrame({
    'text': [x['content'] for x in ds_test_all],
    'labels': [x['label'] for x in ds_test_all]
})

# --- Применяем эвристику для классификации по типу ---
df_train_all['type'] = df_train_all['text'].apply(classify_type_rule)
df_test_all['type'] = df_test_all['text'].apply(classify_type_rule)

type2id = {t:i for i, t in enumerate(sorted(df_train_all['type'].unique()))}
id2type = {i:t for t,i in type2id.items()}
df_train_all['type_id'] = df_train_all['type'].map(type2id)

# type2id = {t:i for i, t in enumerate(df_test_all['type'].unique())}
# id2type = {i:t for t,i in type2id.items()}

df_test_all['type_id'] = df_test_all['type'].map(type2id).fillna(-1).astype(int)

# --- Для ускорения экспериментов берем часть данных ---
frac = 0.25

train_n = int(len(df_train_all) * frac)
test_n = int(len(df_test_all) * frac)

df_train_small = df_train_all.sample(n=train_n, random_state=SEED).reset_index(drop=True)
df_test_small = df_test_all.sample(n=test_n, random_state=SEED).reset_index()

# --- Преобразуем в HuggingFace Datasets ---
hf_train = Dataset.from_pandas((df_train_small[['text', 'labels', 'type_id']]))
hf_test = Dataset.from_pandas((df_test_small[['text', 'labels', 'type_id']]))

# --- Токенизация с обработкой в несколько процессов ---
# NUM_PROC = min(8, max(1, multiprocessing.cpu_count() - 1))
NUM_PROC = 2 
def tokenizer_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

try:
    hf_train = hf_train.map(tokenizer_fn, batched=True, batch_size=2048, num_proc=NUM_PROC)
    hf_test = hf_test.map(tokenizer_fn, batched=True, batch_size=2048, num_proc=NUM_PROC)
except Exception as e:
    print("Warning: parallel map failed, falling back to single process. Error:", e)
    hf_train = hf_train.map(tokenizer_fn, batched=True, batch_size=2048)
    hf_test  = hf_test.map(tokenizer_fn, batched=True, batch_size=2048)

# --- Оставляем только нужные колонки ---
cols_to_keep = ['input_ids', 'attention_mask', 'labels']
remove_cols_train = [c for c in hf_train.column_names if c not in cols_to_keep]
remove_cols_test  = [c for c in hf_test.column_names  if c not in cols_to_keep]
hf_train = hf_train.remove_columns(remove_cols_train)
hf_test  = hf_test.remove_columns(remove_cols_test)

hf_train.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
hf_test.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# --- Настройка модели и тренера ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Загружаем предобученную модель для классификации
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Метрики
accuracy = evaluate.load('accuracy')
f1_macro = evaluate.load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1 = f1_macro.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

# --- Параметры тренировки ---
training_args = TrainingArguments(
    output_dir = './models/sentiment-distilbert_all',
    eval_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    # dataloader_num_workers=min(8, multiprocessing.cpu_count()-1),
    dataloader_num_workers=4,
    num_train_epochs=3,
    logging_steps=200,
    learning_rate=2e-5,
    warmup_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    fp16=torch.cuda.is_available(),  
)

# --- Создаем тренера ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# --- Запускаем тренировку ---
trainer.train()