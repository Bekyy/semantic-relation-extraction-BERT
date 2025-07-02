# semantic-relation-extraction-BERT

## 📌 Project Overview

This project focuses on extracting semantic relation between a pair of word for Amharic language using a fine-tuned pretrained transformer model. The model is trained to learn word relationships, leveraging BERT for classification task.

## 🚀 Features

* Fine-tuned Davlan/bert-base-multilingual-cased-finetuned-amharic for Amharic semantic relation extraction.

* Training on a structured dataset containing Amharic word pairs labeled as synonym,antonyms, homonym and hypernyms.

* Evaluation using  precision, recall, f1-score and accuracy metrics.

* Deployment-ready implementation using Hugging Face Transformers.

## 📂 Dataset

* The dataset consists of three columns:

- word1 - The first pair of Amharic word.

- word2 - The corresponding pair of Amharic word.

* The data is preprocessed and tokenized using Hugging Face's Tokenizer.

## 🏗 Model Training

* The model is trained using AutoModelForTokenClassification with the following setup:
```python
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-finetuned-amharic")
model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-finetuned-amharic", num_labels=4)


# Set training arguments
training_args = TrainingArguments(
    output_dir='./bert-word-semantic-knowledge-cleaned_data_WeightBalanced2',
    evaluation_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    push_to_hub=True,
    report_to="tensorboard",
    save_strategy='epoch',
    save_total_limit=1,  # Keep only the best model based on validation loss
    load_best_model_at_end=True,
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)
```
## 🔥 Inference Example

To generate an antonym for a given Amharic word:
```python
from transformers import pipeline

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-finetuned-amharic")
model = AutoModelForTokenClassification.from_pretrained("Beck90/bert-word-semantic-knowledge-cleaned_data_WeightBalanced2", num_labels=4)

# model.to(device)
# Define the label map
label_map = {
    0: 'synonym',
    1: 'hypernyms',
    2: 'hyponyms',
    3: 'antonyms'
}

# Define the lists of word pairs
word_list1 = ["በላ", "ብርድ", "በረደ", "ሰጠ", "አወራ", "ሄደ", "ሄደ", "ያዘ", "መልካም", "መጥፎ", "አስቀያሚ", "ፍላጎት", "ልባም", "ከፍ", "ፍቅር","ሙሉ","ትልቅ","ቀላል","ሰላም"]
word_list2 = ["ተመገበ", "ቀዝቃዛ", "ቀዘቀዘ", "ለገሰ", "ተናገረ", "መጣ", "ተጏዘ", "ጨበጠ", "ጥሩ", "ጥሩ", "የሚያስጠላ", "አምሮት", "አስተዋይ", "ዝቅ", "ጥላቻ","ባዶ","ትንሽ","ከባድ","ጦርነት"]

# Function to predict the relationship between two words
def predict_relation(word1, word2):
    inputs = tokenizer(word1, word2, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    label_id = predictions[1]  # Label applied to the second token
    label = label_map.get(label_id, "unknown")
    return label

# Iterate through each pair and print the predicted relationship
for w1, w2 in zip(word_list1, word_list2):
    predicted_relation = predict_relation(w1, w2)
    print(f"The predicted relationship between '{w1}' and '{w2}' is: {predicted_relation}")
```
![image](https://github.com/user-attachments/assets/7a93293c-83db-4094-b423-53ddb35895b3)

## Model explainability
* LIME is used to explain the fine-tuned model
```python
# Initialize the LIME explainer
explainer = LimeTextExplainer(class_names=label_map.keys())

# Example text for explanation (replace this with actual samples)
example_texts = ["ፍቅር ጥላቻ", "በላ ተመገበ"]  # Replace with actual word pairs

# Generate explanations
for text in example_texts:
    exp = explainer.explain_instance(text, predict_fn, num_features=10)

    # Show the explanation
    exp.show_in_notebook(text=True)
```
![image](https://github.com/user-attachments/assets/f43c7c62-c31f-4bc3-accc-171735d42adf)

## 📦 Installation & Dependencies
To install the required dependencies:
```bash
pip install transformers torch datasets
```

## ⭐ Acknowledgment

Special thanks to Hugging Face and the Amharic NLP community for their contributions to low-resource language modeling.
