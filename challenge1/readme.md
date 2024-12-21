# **mBART Model Training**

This repository provides a pipeline for training an mBART model using Hugging Face's `Trainer` API for machine translation or similar tasks. The model is fine-tuned on your dataset using the Hugging Face `transformers` library.

## **Table of Contents**
1. [Installation](#installation)
2. [Usage](#usage)
3. [Preprocessing Data](#preprocessing-data)
4. [Training the Model](#training-the-model)
5. [API Documentation](#api-documentation)
6. [Results](#results)
7. [Contributing](#contributing)

# Inference:
```
from transformers import MBartForConditionalGeneration, MBartTokenizer

model = MBartForConditionalGeneration.from_pretrained('path_to_your_trained_model')
tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-50')

input_text = "Your input text here"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs)
decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_text)
```
# Preprocessing Data:
```
from transformers import MBartTokenizer

tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-50')

def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_data = train_data.map(preprocess_function, batched=True)
validation_data = validation_data.map(preprocess_function, batched=True)

```
# Training the model:
```
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data
)

trainer.train()

```
