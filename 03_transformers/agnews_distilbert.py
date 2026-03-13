# %%
import numpy as np

from datasets import load_dataset
from mabt import mabt_ci
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding, 
    Trainer, 
    TrainingArguments
)


# %%
model_name = "distilbert-base-uncased"
n_labels = 4


# %%
raw_data = load_dataset("ag_news")


# %%
train_data = raw_data["train"].shuffle(seed=1).select(range(2000))
test_data = raw_data["test"].shuffle(seed=1).select(range(1000))

print(f"Example training row:\n{train_data[0]}")


# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True)


# %%
train_data = train_data.map(tokenize_batch, batched=True)
test_data = test_data.map(tokenize_batch, batched=True)


# %%
train_data = train_data.rename_column("label", "labels").remove_columns("text")
test_data = test_data.rename_column("label", "labels").remove_columns("text")


# %%
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=n_labels
)


# %%
def accuracy(test_pred):
    logits, labels = test_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_args = TrainingArguments(
    output_dir="./outputs", 
    eval_strategy="epoch", 
    save_strategy="no", 
    logging_strategy="epoch", 
    num_train_epochs=1, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=32, 
    learning_rate=2e-5, 
    weight_decay=0.01, 
    report_to="none", 
    disable_tqdm=True
)

trainer = Trainer(
    model=model, 
    args=train_args, 
    train_dataset=train_data, 
    eval_dataset=test_data, 
    processing_class=tokenizer, 
    data_collator=data_collator, 
    compute_metrics=accuracy
)


# %%
print("Start training...")
trainer.train()


# %%
print("Evaluate on test set...")
test_metrics = trainer.evaluate()
test_acc = test_metrics["eval_accuracy"]
print(f"Test accuracy: {test_acc:.4f}")
# %%
