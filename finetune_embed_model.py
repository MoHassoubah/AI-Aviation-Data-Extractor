from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset, load_dataset
# from torch.utils.data import Dataset
import random
import pandas as pd

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import EvalPrediction



# class SinglePositiveDataset(Dataset):
#     def __init__(self, df):
#         self.df = df
#         # group positives and negatives by query
#         self.pos_map = {}
#         self.neg_map = {}
#         for _, row in df.iterrows():
#             q = row["query"]
#             self.pos_map.setdefault(q, []).append(row["pos_context"])
#             self.neg_map.setdefault(q, []).append(row["neg_context"])
#         self.queries = list(self.pos_map.keys())

#     def __len__(self):
#         return len(self.queries)

#     def __getitem__(self, idx):
#         q = self.queries[idx]
#         # sample one positive and one negative per query
#         pos = random.choice(self.pos_map[q])
#         neg = random.choice(self.neg_map.get(q, [None]))
#         return {"query": q, "pos_context": pos, "neg_context": neg}


# def build_query_map_dataset(df):
#     pos_map = {}
#     neg_map = {}
#     for _, row in df.iterrows():
#         q = row["query"]
#         pos_map.setdefault(q, []).append(row["pos_context"])
#         neg_map.setdefault(q, []).append(row["neg_context"])

#     data = []
#     for q in pos_map:
#         data.append({
#             "query": q,
#             "pos_contexts": pos_map[q],
#             "neg_contexts": neg_map.get(q, [])
#         })

#     return Dataset.from_list(data)

# def dynamic_collate(batch):
#     result = {
#         "query": [],
#         "pos_context": [],
#         "neg_context": [],
#     }
#     for item in batch:
#         result["query"].append(item["query"])
#         result["pos_context"].append(random.choice(item["pos_contexts"]))
#         result["neg_context"].append(random.choice(item["neg_contexts"]) if item["neg_contexts"] else None)
#     return result

# class DynamicCollator:
#     def __init__(self):
#         self.valid_label_columns = []  # Required by SentenceTransformerTrainer

#     def __call__(self, batch):
#         result = {
#             "query": [],
#             "pos_context": [],
#             "neg_context": [],
#         }
#         for item in batch:
#             result["query"].append(item["query"])
#             result["pos_context"].append(random.choice(item["pos_contexts"]))
#             result["neg_context"].append(
#                 random.choice(item["neg_contexts"]) if item["neg_contexts"] else ""
#             )
#         return result

def build_flat_query_dataset(df, seed=None):
    if seed is not None:
        random.seed(seed)
    pos_map = {}
    neg_map = {}
    for _, row in df.iterrows():
        q = row["query"]
        pos_map.setdefault(q, []).append(row["pos_context"])
        neg_map.setdefault(q, []).append(row["neg_context"])

    data = []
    for q in pos_map:
        pos_contexts = pos_map[q]
        neg_contexts = neg_map.get(q, [])
        data.append({
            "query": q,
            "pos_context": random.choice(pos_contexts),
            "neg_context": random.choice(neg_contexts) if neg_contexts else ""
        })

    return Dataset.from_list(data)



model_name = 'all-MiniLM-L6-v2'#"sentence-transformers/all-distilroberta-v1" # acc = 0.88
model = SentenceTransformer('all-MiniLM-L6-v2')#SentenceTransformer(model_name)


df = pd.read_csv("embedding_dataset.csv")


dataset = build_flat_query_dataset(df)#load_dataset('csv', data_files='embedding_dataset.csv')
# train_dataset = SinglePositiveDataset(df)
# print(dataset)

evaluator_valid = TripletEvaluator(
    anchors=dataset["query"],
    positives=dataset["pos_context"],
    negatives=dataset["neg_context"],
    name="ai-job-validation",
)
print("Validation 1:", evaluator_valid(model))

loss = MultipleNegativesRankingLoss(model)

num_epochs = 5000
batch_size = 16
lr = 5e-5
finetuned_model_name = "aviation-finetuned-all-MiniLM-L6-v2"

train_args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{finetuned_model_name}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # ensures that the queries in the batch are unique
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
)


class ResampleDatasetCallback(TrainerCallback):
    def __init__(self, trainer, df):
        self.trainer_ref = trainer
        self.df = df

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"🔁 Resampling dataset at start of epoch {state.epoch}")
        new_dataset = build_flat_query_dataset(self.df, seed=int(state.epoch or 0))
        self.trainer_ref.train_dataset = new_dataset
        self.trainer_ref._train_dataloader = None  # forces dataloader rebuild


trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    loss=loss,
    evaluator=evaluator_valid,
)

trainer.add_callback(ResampleDatasetCallback(trainer, df))


trainer.train()

# evaluator_test = TripletEvaluator(
#     anchors=eval_queries,
#     positives=eval_pos,
#     negatives=eval_neg,
#     name="ai-job-validation",
# )
print("Validation 2:", evaluator_valid(model))
# print("Test:", evaluator_test(model))