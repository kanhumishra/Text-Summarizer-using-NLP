import os 
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset,load_from_disk
from textSummarizer.entity import DataTransformationConfig




class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # 1. Encode the source dialogue inputs
        input_encodings = self.tokenizer(
            example_batch['dialogue'], 
            max_length=1024, 
            truncation=True
        )

        # 2. Encode the target summaries using the updated text_target argument
        target_encodings = self.tokenizer(
            text_target=example_batch['summary'], 
            max_length=128, 
            truncation=True
        )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        # Map tokens across train, test, and validation splits safely
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
