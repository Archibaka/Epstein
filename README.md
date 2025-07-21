<a id="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Archibaka/Epstein">
  </a>

<h3 align="center">Epstein</h3>

  <p align="center">
    100% Local RAG system handling pdf files
    <br />
    <a href="https://github.com/Archibaka/Epstein"><strong>Explore the docs</strong></a>
    <br />
    <a href="https://github.com/Archibaka/Epstein/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/Archibaka/Epstein/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Files, functionality and usage</a></li>
    <ul>
        <li><a href="#Lilim">Lilim.py</a></li>
        <li><a href="#occupy">Occu.py</a></li>
             <ul><li><a href="#test">test.py</a></li>
             </ul>
      <li><a href="#Ser">Ser.py</a></li>
<li><a href="#Main">main.py</a></li>
      <li><a href="#Norm">toNormal.py</a></li>
      <li><a href="#conv">conv.py</a></li>
      </ul>   
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

### Built With

* [![Haystack][Haystack-badge]][Haystack-url]
* [![LangChain][LangChain-badge]][LangChain-url]
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Transformers][Transformers-badge]][Transformers-url]
* [![OpenAI][OpenAI-badge]][OpenAI-url]
* [![HuggingFace][HuggingFace-badge]][HuggingFace-url]
* [![ChromaDB][ChromaDB-badge]][ChromaDB-url]
* [![ONNX][ONNX-badge]][ONNX-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

<!-- Prerequisites -->
### Prerequisites

Install all dependencies from file
* pip
  ```
  pip install -r installPls.txt
  ```

### Installation
  Clone the repository by
  ```
  git clone https://github.com/Archibaka/Epstein.git
  ```
  Create the folder models, inside of it create two folders:
  embmodel for ru-en-RoSBERTa
  and 
  Jeffry for Qwen3
  
  for <a href=https://huggingface.co/ai-forever/ru-en-RoSBERTa/tree/main> ru-en-RoSBERTa </a>  run
  ```
  git lfs install
  git clone https://huggingface.co/ai-forever/ru-en-RoSBERTa /models/embmodel/RoSBERTa
  ```

  
  for <a href=https://huggingface.co/Qwen/Qwen3-1.7B>QWEN3 1.7B</a> run
  ```
  git clone https://huggingface.co/Qwen/Qwen3-1.7B /models/Jeffry/qwen3
  ```

  You might need to use proxy or vpn to successfully clone these in certain countries

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Files, functionality and usage <a id="usage"></a>

Every module could be lauched by itself using
```
python module.py
```

### Lilim.py <a id="Lilim"></a>
Wrapper for LLM (default: <a href=https://huggingface.co/Qwen/Qwen3-1.7B>QWEN3 1.7B</a>, that is accsessible by this path)
  
  Uses <a href="https://huggingface.co/docs/transformers/en/index">huggingface transformers </a> and <a href=https://pytorch.org>torch</a>
  ```
  path = "./models/Jeffry/qwen3"
  ```
The initial prompt is stored in variable 
  ```
  prompthatbitch = "Вы — помощник для работы с базой данных..."
  ```
The model should be loaded using
  ```
  llm = Lilim(path)
  llm.load_model()
  ```  
The two ways of generating are avalible, sharing the arguments.

Output string
  
  ```
  generate(self, user_input, max_new_tokens=1024, temperature=0.7, top_p=0.95, 
                 top_k=20, sample=True, min_p=0, 
                 exponential_decay_length_penalty=None, think=False, cache_implementation=None)
  ```
And Token Generator
  ```
  generateSt(self, user_input, max_new_tokens=1024, temperature=0.7, top_p=0.95, 
                 top_k=20, sample=True, min_p=0, 
                 exponential_decay_length_penalty=None, think=False, cache_implementation=None)
  ```

The parameters passed correspond with the ones described in <a href = "https://huggingface.co/docs/transformers/en/main_classes/text_generation"> This article </a>

The context window could be manipulated by these class methods
```
add_to_history(self, role, content)
```
```
clear_history(self)
```

If launched by itself, runs an example:
```
llm = Lilim(path)
    query = "ye;ty ujcn yf hfphf,jnre gj"
    TRANSLATION_DICT = build_translation_dict()
    prompt = (
        "Time to reformulate some queies: Rephrase this query for better document retrieval. "
        "Focus on key entities and relationships. If this doeasn't make sense,"
        look at the version with changed keyboard layout"
        "Keep it concise. Return Only the Augmented Prompt and nothing else\n\n"
        f"Original: {query}, changed layout: {str.translate(query, TRANSLATION_DICT)}\n"
        "Rephrased:"
    )
    llm.load_model()
    requery = llm.generate(
            prompt, 
            think=True,
            exponential_decay_length_penalty=(1000, 1.1),
            sample=False
        )
    print(requery)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Occu.py <a id="occupy"></a>
Module for converting file directory to the vector database and retrieving data from it

Uses <a href="#Lilim">Lilim</a>, <a href="#Ser">Ser.py</a>, <a href=https://github.com/jsvine/pdfplumber> pdfplumber<a/> and <a href="https://haystack.deepset.ai/">haystack</a>

To convert simply run by itself or call
```
ind()
```
That reads data from your list_path directory and converts to the local vector database in chroma_db folder in your project directory

To retrieve data from your database at ./chroma_db call
```
ret(data)
```

#### Test.py <a id="test"></a>
  Another version of occu.py designed to take adventage of the better hardware 
  
  by running max_worker instances of llm agents

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Ser.py <a id="Ser"></a>
Locally hosted <a href=https://huggingface.co/ai-forever/ru-en-RoSBERTa/tree/main> ru-en-RoSBERTa embedder </a> to satisfy dumbass stupid HuggingFaceAPIDocumentEmbedder from <a href="https://haystack.deepset.ai/">haystack</a> because apparently <a href="#Lilim">it's too difficult to load the model from the directory </a>

Uses <a href="https://huggingface.co/sentence-transformers">huggingface sentence transformers </a> and <a href=https://flask.palletsprojects.com/en/stable/>flask</a>

To use simply run by itself
```
python ser.py
``` 
If you wish, you can change your host and port
<a id="Show him its place"></a>
```
app.run(host=urhost, port=urport)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### toNormal.py <a id="Norm"></a>
Programm to ocr the pdfs 

Uses <a href=https://github.com/ocrmypdf/OCRmyPDF>ocrmypdf</a>

put your files in 

```
outpath = "./files/list"
```

and get your ocred files from

```
inpath = "./files/GOOD"
```

or change the variables

To use simply run by itself
```
python toNormal.py
``` 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### main.py <a id="Main"></a>
Chatbot window that ties everything.

Uses all of the above

Requires running
```
python ser.py
``` 
at
```
ip = "localhost"
port = "8000"
```
to configure this, please consult <a href="#Show him its place">this</a>

in the separate window unless don't want to discuss content from the database with your RAG system

### conv.py <a id= "conv" href=https://stackoverflow.com/questions/78010107/how-to-translate-symbols-from-latin-to-cyrillic> changes keyboard layout </a> 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Move ChromaDatabase to the localhost
- [ ] Make Generation Abortable (Add stop fully implement stop generation functionality)
- [ ] Make models switchable
- [ ] Add file protection

See the [open issues](https://github.com/Archibaka/Epstein/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't even bother to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/Archibaka/Epstein/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Archibaka/Epstein" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

No license idk



<!-- CONTACT -->
## Contact

Project Link: [https://github.com/Archibaka/Epstein](https://github.com/Archibaka/Epstein)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Haystack-badge]: https://img.shields.io/badge/Haystack-4A00D0?style=for-the-badge&logo=ai&logoColor=white
[Haystack-url]: https://haystack.deepset.ai/
[LangChain-badge]: https://img.shields.io/badge/LangChain-00A67E?style=for-the-badge&logo=chainlink&logoColor=white
[LangChain-url]: https://python.langchain.com/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Transformers-badge]: https://img.shields.io/badge/Transformers-FFD21F?style=for-the-badge&logo=ai&logoColor=black
[Transformers-url]: https://huggingface.co/docs/transformers
[OpenAI-badge]: https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white
[OpenAI-url]: https://platform.openai.com/
[HuggingFace-badge]: https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black
[HuggingFace-url]: https://huggingface.co/
[ChromaDB-badge]: https://img.shields.io/badge/ChromaDB-1890F1?style=for-the-badge&logo=vectordatabase&logoColor=white
[ChromaDB-url]: https://www.trychroma.com/
[ONNX-badge]: https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white
[ONNX-url]: https://onnx.ai/
