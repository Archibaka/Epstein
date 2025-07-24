#Programm to find, handle and shove pdf-files to the database (The script does so too)
import os
import os.path

from haystack.components.embedders import HuggingFaceAPITextEmbedder, HuggingFaceAPIDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.preprocessors import DocumentPreprocessor
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy
from haystack import Pipeline
from haystack import Document
from haystack.components.writers import DocumentWriter
from lilim import Lilim

import pdfplumber

# embmodel_path = "/home/rerephil/Desktop/Manhattan/Epstein/models/embodel/RoSBERTa"
list_path = "./files/GOOD"
hostip = "http://localhost:8080"

#Transfer text from OCRED pdfs to the document storage native to the database
def extract_text_and_stuff(list_path, checkpoint):
    savepath = "./files/temp"
    plsHelp = Lilim("./models/Jeffry/qwen3", ass=False)
    plsHelp.load_model()
    prompt = """Задача: Удали мусор из текста строго по правилам. Не добавляй никаких комментариев, не анализируй, просто оставь очищенный текст.
    Делай всё буквалтно, даже если текст содержит неполные и бессмысленные данные
Правила удаления мусора:
- Мусор: случайные наборы символов, которые не образуют слов (например, `трепскэхеТ`, `T`, `H`, `С`, `Э` без контекста, `Gh$k=`, `##x@`, `te. ee. e.ee`, `.eee- ee ns`).
- Не мусор: 
   * Слова на любых языках (даже если они вставлены в текст на другом языке)
   * Аббревиатуры (ГОСТ, NRMS, ACH и т.п.)
   * Номера стандартов (например, 24402-88, 10-14189)
   * Даты (например, 06.02.2024)
   * Имена собственные (например, Орлов Федор Леонидович)
   * Технические обозначения (например, 4.х®)
Действуй так:
1. Найди в тексте мусорные вставки (бессмысленные последовательности символов, не образующие слов и не являющиеся допустимыми обозначениями).
2. Удали только эти вставки, оставив вокруг них пробелы и полезный текст без изменений.
3. Сохрани исходное форматирование (разбивку на строки, пункты списка и т.д.).
4. Не добавляй никаких своих слов, комментариев, подписей.
    Обязательные требования:
        Сохраняй полезные данные между фрагментами мусора
        Не разрывай связанные смысловые блоки
        Сохраняй фрагменты на других языках, если они были в оригинале
        Обрабатывай КАЖДУЮ страницу отдельно
        Работай с ЛЮБЫМИ данными — даже неполными и бессмысленными
        Агрессивно удаляй только явный шум
        Сохраняй все пунктуационные конструкции
        Не пытайся "исправлять" текст — только очистка
        При сомнениях оставляй фрагмент без изменений
        Работай строка за строкой без объединения
    Приоритет: максимальная сохранность значимого контента
    Перепиши только текст из файла, НЕ добавляй ниуего от себя
    Примеры обработки:  
        Ввод: Gh$k=Важный (ИУС 9—81).текст##x@ с числами123
        Вывод: Важный (ИУС 9—81).текст с числами123
    Ввод:ГОСТ 24402-88 Телеобработка данных и вычислительные сети. Термины и
    NormaCS 4.x® (NRMS10-14189)
    Орлов Федор Леонидович
    C 01.07.1989 действует, взамен
    06.02.2024 Стр. 1 из 16
    Вывод:ГОСТ 24402-88 Телеобработка данных и вычислительные сети. Термины и
    NormaCS 4.x® (NRMS10-14189)
    Орлов Федор Леонидович
    C 01.07.1989 действует, взамен
    06.02.2024 Стр. 1 из 16
    ЗАПРЕЩЕНО писать вывод или заключение для информации
    Критическая важность: Качество результата напрямую влияет на мою профессиональную репутацию
    The text:"""
    plsHelp.add_to_history("System", prompt)
    list = []
    #For each file in folder
    for filename in os.listdir(list_path):
        with pdfplumber.open(f"{list_path}/{filename}") as pdf:
                print(f"{filename}:\n")
                txt = ""
                #For each page in file
                for i, page in enumerate(pdf.pages):
                    print(f"Page{i+1}\n") #{os.path.isfile(f"{savepath}/{filename}Page{i+1}.txt")} {savepath}/{filename}Page{i+1}.txt"
                    pg = ""
                    #Check if file with written page exist
                    if (not checkpoint) and (not os.path.isfile(f"{savepath}/{filename}Page{i+1}.txt")):
                        #If not, ask model to clean it up
                        print("Hard way")
                        pg = "\n" + plsHelp.generate("new page: \n" + page.extract_text(), max_new_tokens=131072, think=True)
                        if not checkpoint:
                            #And write it down page by page
                            open(f"{savepath}/{filename}Page{i+1}.txt", "x").write(pg)
                    else:
                        #If yes, just copy it
                        print("Easy way, loading from txt")
                        pg = "\n" + open(f"{savepath}/{filename}Page{i+1}.txt", "r").read()
                    txt += pg
        list.append(Document(content=txt))
    return list

#__________________________________COMPONENTS_________________________________________

#Cleaner for extracted data
processor = DocumentPreprocessor(
    extend_abbreviations=False,
    split_by="word",
    split_length=400,  # Adjust based on model's 512-token limit
    split_overlap=150,
    split_threshold=2,
    remove_repeated_substrings=True,
    respect_sentence_boundary=True,
    language="ru+en"
)

#Vector database itself
document_store = ChromaDocumentStore(
    #host=idk for future hosting of database, do not use with persist_path!
    #port = 69420
    persist_path="./chroma_db",  # Local storage
    collection_name="pdf_embeddings"
)

embedder = HuggingFaceAPIDocumentEmbedder(
    api_type="text_embeddings_inference",
    api_params={"url": hostip},
    normalize=True
)

#Text

Quembedder = HuggingFaceAPITextEmbedder(
    api_type="text_embeddings_inference",
    api_params={"url": hostip},
    normalize=True
)

#Writer to the vector database
writ = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

#Retriever from the database
retriever = ChromaEmbeddingRetriever(
    document_store=document_store,
    top_k=4
)

# print(extract_text_and_stuff(list_path))

#________________________________PIPELINES___________________________________________

# Indexing pipeline: The data is cleaned up and written to the database
index = Pipeline()
index.add_component("Preprocessor", processor)
index.add_component("Embedder", embedder)
index.add_component("Writer", writ)
index.connect("Preprocessor", "Embedder")
index.connect("Embedder.documents", "Writer.documents")

def ind():
    
    # Convert
    list = extract_text_and_stuff(list_path, False) #True if you want to extract data from txt
    
    # Preprocess and embed
    index.run(data={"Preprocessor": {"documents": list}})
    print(f"Embeddings stored in ChromaDB at ./chroma_db")

#Retrieving pipeline: The query is embedded and 10 closest neibours are found
retr = Pipeline()
retr.add_component("QueryEmbedder", Quembedder)
retr.add_component("Retriever", retriever)
retr.connect("QueryEmbedder", "Retriever")

def ret(requery):
    return retr.run({"QueryEmbedder":{"text": requery}})

if __name__ == '__main__':
    ind()
