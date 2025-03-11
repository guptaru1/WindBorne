import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
import os

# Initialize wandb for experiment tracking
wandb.init(project="agricultural-llm")

def prepare_crop_yield_data(csv_path):
    """
    Prepare crop yield dataset for training.
    Expected columns: crop_type, weather_data, soil_quality, yield
    """
    df = pd.read_csv(csv_path)
    
    # Combine features into text format
    df['text'] = df.apply(lambda row: f"""
    Crop Type: {row['crop_type']}
    Weather: {row['weather_data']}
    Soil Quality: {row['soil_quality']}
    """.strip(), axis=1)
    
    # Convert yield into categorical labels (Low, Medium, High)
    df['yield_category'] = pd.qcut(df['yield'], q=3, labels=['Low', 'Medium', 'High'])
    
    return df[['text', 'yield_category']]

def prepare_pest_disease_data(csv_path):
    """
    Prepare pest and disease dataset for training.
    Expected columns: crop_type, pest_type, disease_name, severity, conditions
    """
    df = pd.read_csv(csv_path)
    
    # Combine features into text format
    df['text'] = df.apply(lambda row: f"""
    Crop Type: {row['crop_type']}
    Conditions: {row['conditions']}
    Pest Type: {row['pest_type']}
    Disease: {row['disease_name']}
    """.strip(), axis=1)
    
    # Use severity as label
    df['severity_category'] = pd.qcut(df['severity'], q=3, labels=['Low', 'Medium', 'High'])
    
    return df[['text', 'severity_category']]

def combine_datasets(crop_yield_df, pest_disease_df):
    """Combine both datasets and prepare for training"""
    # Combine datasets
    crop_yield_df['dataset_type'] = 'yield'
    pest_disease_df['dataset_type'] = 'pest_disease'
    
    combined_df = pd.concat([crop_yield_df, pest_disease_df], ignore_index=True)
    
    # Create label mapping
    label2id = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    
    # Convert labels to numeric
    combined_df['label'] = combined_df.apply(
        lambda row: label2id[row['yield_category' if row['dataset_type'] == 'yield' else 'severity_category']],
        axis=1
    )
    
    return combined_df[['text', 'label']], label2id

def create_dataset(df):
    """Convert DataFrame to HuggingFace Dataset"""
    return Dataset.from_pandas(df)

def main():
    # Load and prepare datasets
    crop_yield_df = prepare_crop_yield_data('data/crop_yield.csv')
    pest_disease_df = prepare_pest_disease_data('data/pest_disease.csv')
    
    # Combine datasets
    combined_df, label2id = combine_datasets(crop_yield_df, pest_disease_df)
    
    # Split data
    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    
    # Create HuggingFace datasets
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label={v: k for k, v in label2id.items()},
        label2id=label2id
    )
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model("./models/agricultural-llm")
    tokenizer.save_pretrained("./models/agricultural-llm")

if __name__ == "__main__":
    main() 