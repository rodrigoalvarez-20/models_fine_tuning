
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
from wikipedia_downloader import WikiDownloader
import os
#import fastai2


lang = 'es'
name = f'{lang}wiki'

def prepare_corpus():
    # Descargar Wiki en Español
    wkc = WikiDownloader("es", "es_wiki")
    #wkc.download_wiki()
    #wkc.decompress_bz_file()
    #wkc.extract_wiki()
    #docs_path = wkc.split_wiki()

    #wkc.get_one_clean_file("es_wiki/docs")
    # TODO Pendiente para validacion
    wkc.get_one_clean_csv_file("es_wiki/docs")

def load_gpt_model():
    pretrained_weights = 'gpt2'
    tokenizer_en = GPT2TokenizerFast.from_pretrained(pretrained_weights)
    tokenizer_en.pad_token = tokenizer_en.eos_token
    ByteLevelBPE_tokenizer_es_vocab_size = tokenizer_en.vocab_size
    ByteLevelBPE_tokenizer_es = ByteLevelBPETokenizer()
    paths = [ "es_wiki/all_texts_eswiki.txt" ]
    ByteLevelBPE_tokenizer_es.train(files=paths, 
        vocab_size=ByteLevelBPE_tokenizer_es_vocab_size, 
        min_frequency=2, 
        special_tokens=["<|endoftext|>"])
    ByteLevelBPE_tokenizer_es.enable_truncation(max_length=1024)

    ByteLevelBPE_tokenizer_es_rep = 'ByteLevelBPE_tokenizer_es'
    path_to_ByteLevelBPE_tokenizer_es_rep = "es_wiki/" + ByteLevelBPE_tokenizer_es_rep
    if not os.path.exists(path_to_ByteLevelBPE_tokenizer_es_rep):
        os.mkdir(path_to_ByteLevelBPE_tokenizer_es_rep)
        
    ByteLevelBPE_tokenizer_es.save_model(path_to_ByteLevelBPE_tokenizer_es_rep)
    # 3. Import the tokenizer config files in Portuguese into the pre-trained GPT2 Tokenizer
    tokenizer_es = GPT2TokenizerFast.from_pretrained( path_to_ByteLevelBPE_tokenizer_es_rep, pad_token='<|endoftext|>')
    # Get sequence length max of 1024
    tokenizer_es.model_max_length = 1024




if __name__ == "__main__":
    prepare_corpus()
    #load_gpt_model()
