# Import the required packages
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2ForSequenceClassification, GPT2Tokenizer, Trainer, TrainingArguments

import numpy 
from datasets import Dataset

train = pd.read_csv("Tweets.csv", index_col=0)
text_X = train["text"]
y = train["airline_sentiment"]
y = y.replace(["negative", "neutral", "positive"], [0, 1, 2])


# Split in train test
text_X_train, text_X_test, y_train, y_test = train_test_split(
    text_X, y, train_size=0.1, test_size=0.01, random_state=42
)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# Load the pre-trained model
transformer_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=3)

def tokenize_function(examples):
    return tokenizer(examples, padding='max_length', truncation=True, max_length=128)


training_set = pd.concat([text_X_train, y_train], axis=1)

training_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

testing_set = pd.concat([text_X_test, y_test], axis=1)



# Function that transforms a list of texts to their representation
# learned by the transformer.

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)



trainer = Trainer(
    model=transformer_model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=testing_set,
)

trainer.train()

trainer.evaluate()


def get_sentiment(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = transformer_model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return prediction


# for sentence, true_label in zip(text_X_test, y_test):
#     pred = get_sentiment(sentence)

print(f1_score(y_test, [get_sentiment(i) for i in text_X_test], average="macro"))