#импортируем библиотеку streamlit, чтобы запустить код в приложении
import os
os.system('pip install transformers') 
import streamlit as st
#импортируем модель
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_ru_en = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-ru-en") 
tokenizer_ru_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en") 
model_en_ru = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-ru") 
tokenizer_en_ru = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")


st.title("Переводчик с использованием Hugging Face и Streamlit")
#Название, которые будет видно у нас в приложении
text_input = st.text_area("Введите текст для перевода:", value="", height=200)
#Место для ввода текста, для дальнейшего перевода


def translate_text(text, model, tokenizer): 
    input_ids = tokenizer.encode(text, return_tensors="pt") 
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True) 
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True) 
    return translated_text

def translate(request: TranslationRequest): 
    input_text = request.input_text 
    # Определяем язык текста 
    source_lang = "en" if all(ord(c) < 128 for c in input_text) else "ru"

if st.button("Перевести"):
    if source_lang == "en": 
        translated_text = translate_text(text_input, model_en_ru, tokenizer_en_ru) 
    else: 
        translated_text = translate_text(text_input, model_ru_en, tokenizer_ru_en)
    st.write(translated_text[0])
    #Выводится перевод

    