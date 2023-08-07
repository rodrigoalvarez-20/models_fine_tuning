from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, T5ForQuestionAnswering, AutoTokenizer, T5Tokenizer
import json
import torch

class genshin_impact(Dataset):
    def __init__(self, file_path, qa_model):
        self.data = self.load_data(file_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(qa_model)

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        paragraphs = data['data'][0]['paragraphs']
        extracted_data = []
        for paragraph in paragraphs:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']
                start_pos = qa['answers'][0]['answer_start']
                end_pos = qa['answers'][0]['answer_end']
                extracted_data.append({
                    'context': context,
                    'question': question,
                    'answer': answer,
                    'start_pos': start_pos,
                    'end_pos': end_pos
                })
        return extracted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example['question']
        context = example['context']
        answer = example['answer']
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        start_pos = torch.tensor(example['start_pos'])
        end_pos = torch.tensor(example['end_pos'])
        return input_ids, attention_mask, start_pos, end_pos