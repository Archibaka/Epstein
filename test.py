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
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # For progress visualization

import pdfplumber

# embmodel_path = "/home/rerephil/Desktop/Manhattan/Epstein/models/embodel/RoSBERTa"
list_path = "./files/GOOD"
hostip = "http://10.0.0.37:8080"
model_path = "./models/Jeffry/qwen3"
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

def init_worker():
    """Инициализирует модель для каждого воркера"""
    global lilim_worker
    lilim_worker = Lilim(model_path, ass=False)
    lilim_worker.load_model()

#Transfer text from OCRED pdfs to the document storage native to the database
def process_page(page_data):
    """Обрабатывает страницу с использованием глобальной модели воркера"""
    filename, page_num, page_text, checkpoint, savepath = page_data
    output_file = f"{savepath}/{filename}Page{page_num}.txt"
    
    if checkpoint and os.path.isfile(output_file):
        return open(output_file, "r").read()
    
    try:
        # Сбрасываем историю диалога перед каждой страницей
        lilim_worker.conversation_history = [{"role": "System", "content": prompt}]
        cleaned_text = lilim_worker.generate(
            "new page: \n" + page_text, 
            max_new_tokens=131072, 
            think=True, 
            cache_implementation="static",
            stream=False
        )
        if not checkpoint:
            with open(output_file, "w") as f:
                f.write(str(cleaned_text))
        return cleaned_text
    except Exception as e:
        print(f"Error processing {filename} page {page_num}: {str(e)}")
        return page_text  # Fallback to original text

def extract_text_and_stuff(list_path, checkpoint=False):
    savepath = "./files/temp"
    os.makedirs(savepath, exist_ok=True)
    documents = []
    
    # Подготовка данных страниц (без передачи model_path)
    all_pages = []
    for filename in os.listdir(list_path):
        with pdfplumber.open(os.path.join(list_path, filename)) as pdf:
            for i, page in enumerate(pdf.pages):
                all_pages.append((
                    filename, i+1, page.extract_text(), 
                    checkpoint, savepath
                ))


    # Process pages in parallel
    with ProcessPoolExecutor(
        max_workers=2,
        initializer=init_worker  # Инициализируем модель в каждом воркере
    ) as executor:
        results = list(tqdm(
            executor.map(process_page, all_pages),
            total=len(all_pages),
            desc="Processing pages"
        ))
    
    # Combine results per document
    current_file = None
    combined_text = ""
    for (filename, _, _, _, _), text in zip(all_pages, results):
        if filename != current_file:
            if current_file:
                documents.append(Document(content=combined_text))
            current_file = filename
            combined_text = ""
        combined_text += "\n" + text
    
    if current_file:
        documents.append(Document(content=combined_text))
    
    return documents


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