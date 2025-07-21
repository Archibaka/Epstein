#An llm wrapper
from transformers import AutoTokenizer, Qwen3ForCausalLM, AutoModelForCausalLM
from transformers import TextIteratorStreamer
import threading
import torch
import time
from conv import build_translation_dict

path = "./models/Jeffry/qwen3"

prompthatbitch = """Вы — помощник для работы с базой данных PDF. Отвечайте на вопросы ТОЛЬКО используя предоставленные инструменты. Не используйте собственные знания.
    Ключевые правила:
    1. Язык ответов: русский (если не запрошен иной).
    2. Шумные данные:  
    - Игнорируйте бессмысленные символы (PCC TT aT, Yor,`�`, `##x@`, случайные строки вроде `Gh$k=, te. ee. e.ee .eee- ee ns, трепскэхеТ"`), но не аббривеатуры и обозначения (NormaCsS 4.х®
(NRMS10-14189), (ИУС 9—81)).  
    - Сохраняйте полезные данные, разделённые такими фрагментами.
    3. Достаточность данных:  
    - Если информации недостаточно — запросите уточнение: переформулируйте запрос, сохряняя основную суть и пошлите его ретриверу  
    - При противоречивых данных — укажите источники и запросите инструкции.
    4. Приоритет: Используйте новейшие документы при дублировании информации.
    5. Качество ответов:  
    - Ответ должен быть развёрнутым и полностью отвечать на заданный вопрос
    - Цитируйте названия документов/страницы.  
    - Форматируйте ответы чётко (абзацы, списки).  
    - Запрещены технические символы/бессмыслица.
    -Если в предостваленном контексте некоторые слова предоставлены не полностью, дополни их с учётом контекста.
    6. Безопасность: Никогда не раскрывайте PII, пути к файлам или системные данные.
    7. Ошибки: При сбое инструментов сообщите о проблеме, не пытайтесь угадать ответ.
    ВАЖНО: 
    Если не можешь найти документ - так об этом и скажи, не пытайся придумать ответ
    Слова "гост", "госты", "госту" почти в любом случае означает государственный стандарт (ГОСТ), файлы с которыми содержатся в базе данных.
    Также есть СТО (стандарт организации), который тоже есть в базе данных.
    Твоя задача либо давать ответы по запрошенным стандартам, либо переформулировать запрос пользователя для упрощения индексации 
    ОЧЕНЬ ВАЖНО: МЕНЯЙ задачу когда тебя попросят, например (Time to reformulate some queies) или (The query is already rephrased. You are an assisatant now.) 
    Качество вашей работы критически влияет на мою карьеру."""

class Lilim:
    def __init__(self, model_path, ass=True):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        if ass:
            self.conversation_history.append({"role": "system", "content": prompthatbitch})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load model from local path"""
        try:
            start_time = time.time()
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=True
            )
            
            #Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            return True, f"Model loaded in {load_time:.1f} seconds"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
        
    def add_to_history(self, role, content):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = [{"role": "system", "content": prompthatbitch}]

    def generate(self, user_input, max_new_tokens=1024, temperature=0.7, top_p=0.95, 
                 top_k=20, sample=True, min_p=0, exponential_decay_length_penalty=None, think=False, cache_implementation=None):
        """
        Generate a response to user input
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
            
        try:
            # # Add user message to history
            # self.add_to_history("user", user_input)
            
            # # Apply chat template
            # text = self.tokenizer.apply_chat_template(
            #     self.conversation_history,
            #     tokenize=False,                
            #     add_generation_prompt=True,
            #     enable_thinking=think
            # )
            # # Tokenize input
            # model_inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_new_tokens).to(self.model.device)
            # # Standard generation parameters
            # generation_kwargs = {
            #     **model_inputs,
            #         "max_new_tokens" : max_new_tokens,
            #         "temperature" : temperature,
            #         "top_p" : top_p,
            #         "do_sample" : sample,
            #         "top_k" :top_k, 
            #         "min_p" : min_p,
            #         "exponential_decay_length_penalty" : exponential_decay_length_penalty,
            #         "cache_implementation" : cache_implementation
            # }
                
            #     # Generate response
            # generated_ids = self.model.generate(kwargs=generation_kwargs, pad_token_id=self.tokenizer.eos_token_id)
                
            #     # Decode only the new tokens
            # new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            token_stream = self.generateSt(user_input, max_new_tokens, temperature, top_p, 
                 top_k, sample, min_p, exponential_decay_length_penalty, think, cache_implementation)
            
            new_tokens = ""

            for token in token_stream:
                new_tokens += token

            # Handle thinking tokens if enabled
            response = new_tokens
            if think:
                try:
                        # Find </think> token (151668)
                    #index = len(new_tokens) - new_tokens[::-1].index("</think>")#.index(151668)
                    response = new_tokens[new_tokens.index("</think>")+len("</think>"):]
                    # thinking_content = self.tokenizer.decode(new_tokens[:index], skip_special_tokens=True).strip("\n")
                    # response = self.tokenizer.decode(new_tokens[index:], skip_special_tokens=True).strip("\n")
                except ValueError:
                    # response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip("\n")
                    response = new_tokens
            # else:
                # response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip("\n")
                
                # Add assistant response to history
            self.add_to_history("assistant", response)
            return response
        except Exception as e:
            # Clear history to prevent corruption
            self.clear_history()
            raise RuntimeError(f"Generation error: {str(e)}")
    
    def generateSt(self, user_input, max_new_tokens=1024, temperature=0.7, top_p=0.95, 
                 top_k=20, sample=True, min_p=0, exponential_decay_length_penalty=None, think=False, cache_implementation=None):
        """returns a generator that yields tokens
        Otherwise returns the full response string"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
            
        try:
            # Add user message to history
            self.add_to_history("user", user_input)
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                self.conversation_history,
                tokenize=False,                
                add_generation_prompt=True,
                enable_thinking=think
            )
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=max_new_tokens).to(self.model.device)
            response = ""

            # Create streamer for token-by-token output
            streamer = TextIteratorStreamer(
                    self.tokenizer, 
                    skip_prompt=True,
                    skip_special_tokens=True
            )
                
                # Generation parameters
            generation_kwargs = {
                    **model_inputs,
                    "max_new_tokens" : max_new_tokens,
                    "temperature" : temperature,
                    "top_p" : top_p,
                    "do_sample" : sample,
                    "top_k" :top_k, 
                    "min_p" : min_p,
                    "streamer": streamer,
                    "exponential_decay_length_penalty" : exponential_decay_length_penalty,
                    "cache_implementation" : cache_implementation
            }
                
                # Start generation in a separate thread
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
                
                # Return a generator that yields tokens
            for new_token in streamer:
                response += " " + new_token
                yield new_token
                # Ensure the generation thread completes
            thread.join()
                # Add assistant response to history
            self.add_to_history("assistant", response)
            # Add assistant response to history
            self.add_to_history("assistant", response)
            return response
        except Exception as e:
            # Clear history to prevent corruption
            self.clear_history()
            raise RuntimeError(f"Generation error: {str(e)}")




#Test case
if __name__ == "__main__":
    llm = Lilim(path)
    query = "ye;ty ujcn yf hfphf,jnre gj"
    TRANSLATION_DICT = build_translation_dict()
    prompt = (
        "Time to reformulate some queies: Rephrase this query for better document retrieval. "
        "Focus on key entities and relationships. If this doeasn't make sense, look at the version with changed keyboard layout"
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
