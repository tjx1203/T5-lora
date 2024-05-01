from datasets import load_dataset
from transformers import AutoTokenizer  # 通用的分词器类，能够自动加载与指定预训练模型匹配的分词器
from transformers import AutoModelForSeq2SeqLM  # 用于Seq2Seq任务的自动模型类, 将一种形式的序列（如文本、语音等）转换为另一种形式的序列（如翻译、摘要等）
from datasets import concatenate_datasets  # 将多个数据集合并为一个数据集。
import numpy as np

import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# ====================== 加载 samsum 数据集 ============================
dataset = load_dataset('samsum')

print(dataset)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")
'''
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary'],   id  对话  摘要  
        num_rows: 14732
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 819
    })
    validation: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 818
    })
})
Train dataset size: 14732
Test dataset size: 819
'''

# ====================== 加载 分词器 ============================
model_id = 'google/flan-t5-xxl'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ====================== 计算文本和摘要的截断长度：通过对 输入文本和摘要 分词转码， 获取序列长度，然后计算文本和摘要的截断长度============================
# 生成式文本摘要属于文本生成任务。我们将文本输入给模型，模型会输出摘要。
# 我们需要了解输入和输出文本的长度信息，以利于我们高效地批量处理这些数据。

# 分词后最大总输入序列长度    长的截断，短的填充
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]). \
    map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
# ①将训练集train和测试集test合并
# ②map: 匿名函数映射到合并数据集上， 这个匿名函数作用：对合并后的数据集进行分词
#        batched=True：`map`函数按批次处理数据
#        remove_columns=["dialogue", "summary"]：在分词之后，不再需要原始的对话和摘要文本。因此，remove_columns参数指示.map()函数从结果数据集中删除这些列。
# ③匿名函数：对数据集中的['dialogue']进行分词，如果分词后的序列长度超过模型的输入长度限制，则会使用`truncation=True`进行截断。
''' tokenizer【T5TokenizerFast】：对文本分词后，返回的字段：id, [被删除：dialogue， summary], input_ids, attention_mask
input_ids: 文本切token转码的结果，其中码对应token在词表中的数字
attention_mask: 指示哪些输入标记是填充标记（padding tokens）'''

# 每个输入文本分词后的长度【token数量】
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# 取最大长度的85%: np.percentile对会先对input_lengths所有的长度排序，然后找到 85%长度 的位置
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")  # 这个值被当做截断长度：输入文本长度超过该长度就截断

# -------------------以下部分与输入文本的处理类似，但针对的是目标文本（即摘要）。---------------------------------
# 分词后 目标文本【摘要】 最大总序列长度
# 合并train和test，处理一个batch的数据，对summary字段分词，长的截断，短的填充， 最后删除"dialogue", "summary"
# 最后得到的tokenized_targets包括：摘要分词转码的结果，attention_mask
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])

# 计算每个摘要分词后的长度【token数量】
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# 取摘要最大长度的90% 对应的长度值，作为截断的标准
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")


# ====================== 数据预处理：对dataset数据集中的数据【dialogue和summary】分词-转码-填充-截断， 然后将数据集保存在磁盘 ============================
def preprocess_function(sample, padding="max_length"):  # sample：一个batch的数据---{id:list, dialogue:list, summary:list}
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]  # 在每一句dialogue添加前缀"summarize: "，这是T5模型期望的输入格式。

    # tokenize inputs  对dialogue分词转码，同时长于max_source_length则截断，短于padding="max_length"则填充
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    ''' 以上是对 输入 先进行 分词-截断-填充， 此时model_inputs中只有input_ids，attention_mas两个字段'''

    # Tokenize targets with the `text_target` keyword argument
    # 对summary分词转码，同时长于max_source_length则截断，短于padding="max_length"则填充
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)
    ''' 这一句是对 摘要 进行 分词-截断-填充， 此时labels中只有input_ids，attention_mas两个字段'''

    # 如果填充方式设置为"max_length"，则执行以下操作
    if padding == "max_length":
        # 遍历label中每一个句子，遍历每个句子的每个token，判断是否是填充，用-100替换所有的填充标记的ID
        # 在PyTorch和TensorFlow等深度学习框架中，-100通常被用作一个特殊的标记值，告诉模型在计算损失时忽略这个位置的损失。
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    '''此时model_inputs中多了一个labels字段，记录摘要的分词转码结果'''
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 映射preprocess_function函数：实现对dataset中所有数据的dialogue和summary字段分词转码[一个batch的一次处理]，
# 最后移除原始字段，只留下函数返回的字段：input_ids输入文本分词转码结果, attention_mask, labels目标文本分词转码结果
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")  # 打印训练集中所有特征的键：

# 将预处理后的训练集和测试集保存到磁盘上，以便以后轻松加载。
tokenized_dataset["train"].save_to_disk("data/train")
tokenized_dataset["test"].save_to_disk("data/eval")

# ======================== 使用 LoRA FLAN-T5 进行评估和推理 ====================================================================
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ======================== 加载模型 ====================================================================
model_id = 'philschmid/flan-t5-xxl-sharded-fp16'
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')
# 模型将以8位（即半精度）格式加载。这可以加速推理和减少内存使用，但可能会稍微降低精度。

# =================================定义LoRA的超参数，修改预训练模型========================================================
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# get_peft_model（用于在模型上添加LoRA适配器）、prepare_model_for_int8_training（用于准备模型以进行int8训练）和TaskType（定义任务类型的枚举）

# 定义lora参数
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # 缩放因子，用于缩放低秩矩阵的权重。
    target_modules=['q', 'v'],
    lora_dropout=0.05,
    bias='none',  # 是否更新偏差
    task_type=TaskType.SEQ_2_SEQ_LM  # 定义任务类型:序列到序列的语言模型任务（`SEQ_2_SEQ_LM`）
)

# 上文已定义预训练模型，这一步是准备模型进行int8训练
model = prepare_model_for_int8_training(model)

# 给模型添加lora参数: 在训练过程中只更新LoRA层（即低秩矩阵和相关的缩放因子）
model = get_peft_model(model, lora_config)

# 打印模型所有可训练参数
model.print_trainable_parameters()

# ===================== 创建一个 DataCollator，负责对输入和标签进行填充=======================
# 批量处理输入数据和标签
from transformers import DataCollatorForSeq2Seq

# 计算损失时忽略填充的token
label_pad_token_id = -100

# 数据收集器
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8  # 指定了输出张量的长度应该是8的倍数。这通常是为了优化硬件性能，因为某些硬件（如GPU）在处理具有特定长度的张量时性能更好。
)

# =================== 定义训练超参 =============================================
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

output_dir = "lora-flan-t5-xxl"  # 输出目录: 模型训练结果和日志

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,  # 自动为训练找到最佳批处理大小
    learning_rate=1e-3,  # higher learning rate
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",  # 日志文件的保存路径
    logging_strategy="steps",  # 日志记录策略，这里设置为“steps”，表示按步骤记录日志。
    logging_steps=500,  # 每500步记录一次日志
    save_strategy="no",  # 保存策略，这里设置为“no”，表示不保存模型检查点。这通常用于调试或快速迭代，但在实际训练中，你可能希望保存模型检查点。
    report_to=["tensorboard"]  # 表示将日志报告给TensorBoard进行可视化
)

# 创建Trainer实例
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,  # 数据整理器，用于将数据整理成模型可以接受的格式。
    train_dataset=tokenized_dataset["train"],
)

# 关闭模型缓存
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
# 禁用了模型的缓存功能，以减少内存使用并消除某些与缓存相关的警告。然而，在推理（即模型预测）时，应该重新启用缓存功能，以提高性能。

# ======================== 训练模型 ============================================
trainer.train()

# 保存LoRA微调后的模型 & 保存分词器， 保存目录：result
peft_model_id = "results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)


# ====================== 使用 LoRA FLAN-T5 进行评估和推理 ==============================
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 加载peft配置
peft_model_id = "results"
config = PeftConfig.from_pretrained(peft_model_id)

# 加载基础大模型（LLM）和分词器
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map={"": 0})
# 从`PeftConfig`中获取基础模型的名称或路径； 以8位量化形式加载模型，以减少内存使用； 指定所有参数都加载到第一个设备，通常是GPU 0

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 加载LoRA模型:
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"": 0})
model.eval()  # 将模型设置为评估模式

print("Peft model loaded")  # 打印加载完成的消息


# ===================== 用测试数据集中的一个随机样本来试试摘要效果========================
print('\n\n===================== 用测试数据集中的一个随机样本来试试摘要效果========================')
from datasets import load_dataset
from random import randrange

# 加载"samsum"的数据集 and 从测试集中随机选择一个样本
dataset = load_dataset("samsum")
sample = dataset['test'][randrange(len(dataset["test"]))]

# 提取样本中的对话，分词转码：长则截断，返回tensor, 移动到GPU上
input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
# 使用模型生成摘要: 生成最多10个新的tokens, 使用采样而不是贪婪解码来生成摘要, 采样时使用的top-p参数
outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)

# 打印输入的对话
print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

# 模型生成的输出ID解码为文本：从GPU上获取输出张量，将其从PyTorch张量转换为NumPy数组，并确保不会计算梯度；在解码时跳过特殊tokens（如填充或分隔符）。
# 从解码的摘要列表中获取第一个（也是唯一的）摘要，因为我们只生成了一个摘要。
print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")