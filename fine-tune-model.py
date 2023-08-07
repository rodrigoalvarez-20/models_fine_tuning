from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, T5ForQuestionAnswering, AutoTokenizer, T5Tokenizer
from GPUtil import showUtilization as gpu_usage
from numba import cuda
from gi_model import genshin_impact
from transformers import pipeline
import json
import torch
import os

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8032"

#qa_model = 'mrm8488/spanish-t5-small-sqac-for-qa'
qa_model = "IIC/roberta-base-spanish-sqac"

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

file_path = 'gi_sqac_ds/gi_answers.json'
dataset = genshin_impact(file_path, qa_model)

free_gpu_cache()

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda")
model = RobertaForQuestionAnswering.from_pretrained(qa_model)
#model = T5ForQuestionAnswering.from_pretrained(ckpt).to(device)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 10
num_epochs = 2
model.gradient_checkpointing_enable()
# Create data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in data_loader:
        # Move batch tensors to the device
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        start_positions = batch[2].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions)
        
        #print(outputs)
        loss = outputs.loss
        
        # Backward pass and optimization
        if loss:
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

model.save_pretrained("fine_trained/gi_roberta_sqac")

#gi_model = RobertaForQuestionAnswering.from_pretrained()

#nlp = pipeline("document-question-answering", model="./fine_trained/gi_roberta_sqac")
#text = "¿Cuál es el nombre de la Shogun Raiden en japonés?"
#context = "==Shogun Raiden==\n\n==Info==\nLa Shogun Raiden (en japonés: 雷電 将軍 Raiden Shougun), cuyo nombre real es Ei, es un personaje jugable en Genshin Impact.\nElla controla una marioneta mientras medita dentro del Plano de la eutimia. Ella es el recipiente mortal de Beelzebul, la actual Arconte Electro de Inazuma.\nHizo su primera aparición como PNJ en la Versión 2.0 y llegó por primera vez en la Versión 2.1 en el gachapón \"Reino de la serenidad\"."
  
#qa_results = nlp(text, context)
#print(qa_results)