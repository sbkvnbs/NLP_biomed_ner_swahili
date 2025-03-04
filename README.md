# NLP_biomed_ner_swahili
NLP project-Biomedical ner for Swahili

##Named Entity Recognition (NER) Model with Transformers

#Overview
This project implements a Named Entity Recognition (NER) model using the Hugging Face Transformers library. The model is trained on tokenized text data and evaluated on a test dataset. It leverages pre-trained transformer models for token classification and uses the Weights & Biases (W&B) tool for experiment tracking.
Requirements
Ensure you have the following dependencies installed:
pip install torch transformers datasets scikit-learn wandb

#Dataset Preparation
The dataset is loaded as a DatasetDict from Pandas DataFrames:
•	train_df → Training dataset
•	val_df → Validation dataset
•	test_df → Test dataset
Each dataset is converted into a Hugging Face Dataset format.
Tokenization
The dataset is tokenized using a tokenizer function that:
•	Tokenizes the Name column.
•	Aligns labels (NER_Category) with tokenized inputs.
•	Ensures padding and truncation up to 128 tokens.
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["Name", "NER_Category"])

#Model Training
A pre-trained model is loaded for token classification:
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
Training parameters are set using TrainingArguments, specifying:
•	Learning rate, batch sizes, weight decay, and training epochs.
•	Mixed precision training (fp16=True).
•	W&B integration for logging.
The model is trained using the Trainer class:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer
)
Training is initiated using:
trainer.train()

#Model Evaluation
After training, the model is evaluated on the validation set:
trainer.evaluate()
To evaluate the model on the test dataset:
•	The test set is tokenized and converted to tensors.
•	Predictions and ground truth labels are extracted.
•	Labels are mapped back to NER tags, and B-/I- prefixes are removed.
•	Accuracy, Precision, Recall, and F1-score are computed using scikit-learn.
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average="macro")
recall = recall_score(labels, predictions, average="macro")
f1 = f1_score(labels, predictions, average="macro")
Logging with Weights & Biases (W&B)
To track training progress, initialize W&B:
wandb.init(project="NER-Project")

#Results
After training and evaluation, the final metrics are printed:
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
Conclusion
This project implements an end-to-end NER model using transformers, allowing efficient token classification with pre-trained models. The evaluation metrics help assess model performance on unseen data.
