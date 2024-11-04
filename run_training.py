import json
import torch
import evaluate
import pandas as pd
from datasets import Dataset
import sentencepiece as spm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence

from models.fnet import FNetForPreTraining

with open('models/fnet_pt_checkpoint/config.json', 'r') as f:
    config = json.load(f)

train_df = pd.read_csv('data/sst2/train.tsv', delimiter='\t', header=0)
validation_df = pd.read_csv('data/sst2/dev.tsv', delimiter='\t', header=0)
test_df = pd.read_csv('data/sst2/test.tsv', delimiter='\t', header=0)


# 转换为 HuggingFace Datasets 格式
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = spm.SentencePieceProcessor(model_file='models/c4_bpe_sentencepiece.model')

# 数据预处理函数
def preprocess_function(examples):
    input_ids = [tokenizer.EncodeAsIds(text) for text in examples['sentence']]
    return {'input_ids': input_ids, 'label': examples['label']}

def preprocess_test_function(examples):
    input_ids = [tokenizer.EncodeAsIds(text) for text in examples['sentence']]
    return {'input_ids': input_ids, 'index': examples['index']}  # 仅添加 'index' 字段，没有标签

train_dataset = train_dataset.map(preprocess_function, batched=True)
validation_dataset = validation_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_test_function, batched=True)

class CustomDataCollator:
    def __call__(self, features):
        input_ids = [torch.tensor(f['input_ids']) for f in features]
        max_length = config['max_position_embeddings']

        # 填充序列，使用配置文件中的 pad_token_id
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=config['pad_token_id'])
        input_ids_padded = input_ids_padded[:, :max_length]

        type_ids = [torch.tensor([f['label']] * input_ids_padded.size(1)) for f in features]
        type_ids = torch.stack(type_ids).to(torch.int64)
        batch = {'input_ids': input_ids_padded, 'type_ids': type_ids}
        return batch

data_collator = CustomDataCollator()

fnet = FNetForPreTraining(config)
fnet.load_state_dict(torch.load('models/fnet_pt_checkpoint/fnet_for_pretraining.statedict.pt', weights_only=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fnet.to(device)

fnet.train()
# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,  # 每500步进行一次评估
    logging_dir="./logs",  # 日志目录
    logging_steps=100,  # 每100步记录一次日志
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
)

# 自定义评估函数
accuracy_metric = evaluate.load("./metrics/accuracy")

def compute_metrics(p):
    preds = torch.argmax(p.predictions["nsp_logits"], dim=-1)
    return accuracy_metric.compute(predictions=preds, references=p.label_ids)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # 获取输入参数
        input_ids = inputs["input_ids"].to(device)
        type_ids = inputs["type_ids"].to(device)
        mlm_positions = inputs.get("mlm_positions", None)
        if mlm_positions is not None:
            mlm_positions = mlm_positions.to(device)

        # 调用模型的 forward 方法
        outputs = model(input_ids=input_ids, type_ids=type_ids, mlm_positions=mlm_positions)

        nsp_logits = outputs["nsp_logits"]
        type_ids = type_ids[:, 0].unsqueeze(-1)

        # 使用适当的损失函数（例如 CrossEntropyLoss）
        loss_fct = torch.nn.CrossEntropyLoss()
        type_ids = type_ids.view(-1,1)
        # 计算 NSP 损失
        nsp_loss = loss_fct(nsp_logits.view(-1,2), type_ids.view(-1))

        # 在每个训练步骤计算并打印准确率
        preds = nsp_logits.argmax(-1)
        accuracy = accuracy_score(type_ids.view(-1).cpu().numpy(), preds.cpu().numpy())
        print(f"Accuracy: {accuracy}")
        return (nsp_loss, outputs) if return_outputs else nsp_loss
    

# 初始化Trainer
trainer = CustomTrainer(
    model=fnet,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()
model_save_path = "models/fnet_sst2/fnet_sst2.statedict.pt"
torch.save(fnet.state_dict(), model_save_path)

def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    index = [item['index'] for item in batch]

    # 使用 pad_sequence 进行填充
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {'input_ids': input_ids_padded, 'index': torch.tensor(index)}

test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

# 生成预测结果并保存到TSV文件
output_file = 'results/predictions.tsv'
with open(output_file, 'w') as f:
    f.write('id\tlabel\n')  # 写入表头
    for batch in test_dataloader:
        input_ids = batch['input_ids']
        idx = batch['index']
        
        with torch.no_grad():
            outputs = fnet(input_ids=input_ids)
        
        preds = torch.argmax(outputs['nsp_logits'], dim=1)
        for i in range(len(idx)):
            f.write(f"{idx[i]}\t{preds[i].item()}\n")

print(f"预测结果已保存到 {output_file}")