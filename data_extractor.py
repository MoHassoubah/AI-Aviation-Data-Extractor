from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_aws import BedrockEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import shutil
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
print("*******************************")
print(OLLAMA_BASE_URL)
from typing import List
from fastapi import UploadFile, File
# from io import BytesIO
import tempfile

import requests
import json
import time

import re

from transformers import AutoTokenizer

import string
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer # Import Porter Stemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
# --- Initialize Stemmer ---
stemmer = PorterStemmer()

from PyPDF2 import PdfReader
import pdfplumber

    
from rank_bm25 import BM25Okapi


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
(Note: Context consists multiple sections seperated by \n\n---\n\n. Each section starts with a score number, the Name of the source document of the context then the Section content)

{context}

---

{question}
"""


CHROMA_PATH = "./chroma"


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
        # region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text",base_url=OLLAMA_BASE_URL)
    return embeddings
     

def generate_response(prompt, model='llama3.1'):
    url = OLLAMA_BASE_URL+"/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "options": {"num_ctx": 60000} 
    }
    response = requests.post(url, json=data)
    print(response)
    return json.loads(response.text)['response']
    





def extract_json_frm_string(json_str):
    
    try:
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            json_part = match.group()
            # Remove // comments from the JSON (inline or standalone)
            json_part = re.sub(r'//.*?(?=\n|$)', '', json_part)
            # Optional: remove trailing commas (if any)
            json_part = re.sub(r',\s*([\}\]])', r'\1', json_part)
             # Fix malformed entries: {"Some Text"} ? {"Some Text": ""}
            # This handles entries like: {"Single Radio Altimeter"}
            fixed_json_part = re.sub(r'\{\s*"([^"]+)"\s*\}', r'\1', json_part)
            fixed_json_part = re.sub(r'\{\s*((?:"[^"]+",?\s*)+)\s*\}', r'\1', fixed_json_part)
            data = json.loads(fixed_json_part)
            if isinstance(data, dict):
                return data
            else:
                return {}
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from: {fixed_json_part} | Error: {e}")
        
    return {}


def get_tokenizer(model="D:\\gen_ai_d\\Meta-Llama-3.1-8B-Instruct"):
    # Load tokenizer (you can use 'meta-llama/Llama-3-8B-Instruct' if available)
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer
    
def count_tokens(text):
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))
        
LLAMA_31_MAX_TOKENS = 8192
LLAMA_31_RESPONSE_TOKENS = 512
def truncate_context(context, max_tokens=LLAMA_31_MAX_TOKENS, response_tokens=LLAMA_31_RESPONSE_TOKENS):
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(context)
    print(f"Number of TOKEN in the PROMPT = {len(tokens)}")
    if len(tokens)  + response_tokens > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

##################################################

def extract_text_table_from_pdf(pdf_path):

    def format_table(table):
        """Format a table as a readable string"""
        if not table or len(table) < 2:
            return ""
        
        headers = table[0]
        lines = []
        for row in table[1:]:
            line_parts = []
            for i, cell in enumerate(row):
                if headers[i]:
                    line_parts.append(f"{headers[i].strip()}: {cell.strip() if cell else ''}")
            lines.append(", ".join(line_parts))
        return "\n".join(lines)

    output_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # print(f"Processing page {page_num}...")
            
            # Extract all text with positions
            words = page.extract_words()
            text_blocks = sorted(words, key=lambda x: (x['top'], x['x0']))
            
            # Group text by lines (basic line reconstruction)
            lines = []
            current_line = []
            last_top = None

            for word in text_blocks:
                if last_top is None or abs(word['top'] - last_top) < 5:
                    current_line.append(word['text'])
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word['text']]
                last_top = word['top']
            if current_line:
                lines.append(" ".join(current_line))

            # Add reconstructed lines to output
            output_text += "\n".join(lines) + "\n\n"

            # Extract tables and add formatted versions
            tables = page.extract_tables()
            for table in tables:
                formatted = format_table(table)
                output_text += "[Extracted Table]\n" + formatted + "\n\n"

    
    return output_text

 

def extract_text_from_pdf(pdf_path):
    # reader = PdfReader(pdf_path)
    # text = ""
    # for page in reader.pages:
        # text += page.extract_text() + "\n"
    # return text
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

        
def split_into_chunks(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def hybrid_search(query, bm25, model, index, chunks, chunk_map, top_k=5, alpha=0.5):
    # BM25
    # bm25_scores = bm25.get_scores(query.lower().split())
    
    # FAISS
    query_vec = model.encode([query], convert_to_numpy=True)
    # query_vec = np.array(model.embed_documents([query]), dtype='float32')
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, len(chunks))
    faiss_scores = np.zeros(len(chunks))
    faiss_scores[I[0]] = D[0]
    
    # Combine
    scores = faiss_scores#alpha * bm25_scores + (1 - alpha) * faiss_scores
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return [(chunks[i], chunk_map[i], scores[i]) for i in top_indices]



stop_words = set(stopwords.words('english'))
   
# def clean_tokens(text):
#     return {word.strip(string.punctuation).lower() for word in text.split()}

def extract_context(files: List[UploadFile], chunk_size):
# Example for multiple PDFs
    documents = []
    filenames = []
    for file in files:
        # content = BytesIO(file.file.read())  # Turn UploadFile into file-like object
        suffix = os.path.splitext(file.filename)[1]  # Get .pdf extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name
            text = extract_text_table_from_pdf(temp_file_path)
            documents.append(text.lower())
            filenames.append(file)

            
    chunks = []
    chunk_map = []

    for i, doc in enumerate(documents):
        doc_chunks = split_into_chunks(doc,chunk_size)
        # print(f"original doc {doc}")
        # print("----------------------------------")
        # print(f"doc chunks{doc_chunks}")
        # input()
        chunks.extend(doc_chunks)
        chunk_map.extend([(filenames[i], j) for j in range(len(doc_chunks))])
        
    print(f">>>>>>>>>>>>>size of chunks = {len(chunks)}")
    # tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    # bm25 = BM25Okapi(tokenized_corpus)
    bm25=None
    
    # model = SentenceTransformer('./models/aviation-finetuned-all-MiniLM-L6-v2/checkpoint-5000')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # model = get_embedding_function()
    # # For embedding a list of chunks (documents):
    # embeddings = np.array(model.embed_documents(chunks), dtype='float32')

    # Normalize if using cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product for cosine
    index.add(embeddings)
    
    
    
    return bm25,model,index,chunks, chunk_map,embeddings

#####################################
def query_pdf(files: List[UploadFile]):

    instructions = """
      "system": "You are an expert aircraft data extraction system. Based only on the above context Return results in strict JSON format with structured keys exactly as described. Apply all rules strictly.",
      "instructions": {
        "output_format": "Strict JSON with structured keys and values only.",
        "line_by_line": "Each parameter as its own JSON key-value pair.",
        "dates": "Use DD/MM/YYYY format for all dates. If only month and year are available, default to DD = 01.",
        "time_cycles": "Numbers only, no units.",
        "detail_fields": "Compress lists into single strings separated by ; if needed.",
        "special_date_rules": {
          "camps": "Extract first 'as of [DATE]' on page 1, apply to all as_of_date fields.",
          "jssi": "Find serial number s/n on page 1, extract the date immediately to its right, apply to all as_of_date fields.",
          "other": "Only use explicitly labeled dates."
        },
        "version_control": {
          "compare_to_previous": "PREVIOUS_STATE",
          "increase": "Mark as CURRENT and update values.",
          "decrease": "Mark as OUTDATED and keep previous values.",
          "first_analysis": "Mark as BASELINE."
        },
        "data_handling": {
          "service_date_synonyms": "Certification Date; Manufactured Date; In-Service Date",
          "date_conversions": {
            "yyyy": "01/01/YYYY",
            "yyyy_yyyy": "01/01/[MostRecentYear]",
            "month_dd_yyyy": "DD/MM/YYYY"
          },
          "document_type_detection": {
            "camps": "if contains 'CAMPs' or 'Continuous Airworthiness'",
            "jssi": "if contains 'JSSI' or 'Jet Support Services'",
            "other": "Default"
          },
        },
      
    }
    """
    Hard_Rule= "Return the required response exactly as provided in required_response. No markdown, No commentary, and No field names beyond the keys in required_response. Do not return any 'Field' or 'Value' pairs - instead use keys directly with their values."
    scheduled_maintenance_logic = """{
            "reference_date": "entryToServiceDate",
            "recognize_titles": [
              "012 Month 00500 Hours",
              "024 Month 01000 Hours",
              "048 Month 02000 Hours",
              "072 Month 03000 Hours",
              "096 Month 04000 Hours",
              "144 Month 06000 Hours",
              "192 Month 08000 Hours",
              "288 Month 12000 Hours",
              "Gear 144 Month Overhaul"
            ],
            "key_format": "Use underscore format e.g. '012M_0500H', 'Gear_144M_Overhaul'",
            "estimate_due_date_if_missing": "Calculate dueDate as entryToServiceDate + X months when not provided",
            "use_performed_date_if_present": "If actual performedDate or dueDate is present, use it directly"
          }"""
          
          
    engines= "Store separate entries for each engine in a JSON array with objects."
    required_response = [
            # make, model, serialNumber, registration, manufacturer>> from all documents
            "make: str = Field(description= Make of the aircraft or the plane)",
            "model: str = Field(description= Model of the aircraft or the plane)",
            "serialNumber: str = Field(description= s/n Serial Number specific to this aircraft or the plane. It's only number no letters.)",
            "registration: str = Field(description= Registration number of the aircraft or the plane. It comes after the serial number by |)",
            "manufacturer: str = Field(description= Manufacturer name of the aircraft or the plane)",
            #mainly from the spec sheet
            "airframeProgram:  str = Field(description= Airframe Power-by-The-Hour Plan of the aircraft or the plane)",
            #totalTimeHours,totalCycles>> all documents (take the largest number)
            "totalTimeHours: int = Field(description= total Flight Time in Hours of the aircraft or the plane. If there are multiple correspoinding values,take the largest value.)",
            "totalCycles: int = Field(description= total Flight Time in Cycles of the aircraft or the plane. If there are multiple correspoinding values,take the largest value.)",
            #flightHoursAsOfDate, entryToServiceDate>> from camps and jssi, (take the lastest date for flightHoursAsOfDate)
            "flightHoursAsOfDate: DateTime = Field(description= flight Hours As Of Date DD/MM/YYYY of the aircraft or the plane)",
            "entryToServiceDate: DateTime = Field(description= Date DD/MM/YYYY of the First Service of the aircraft or the plane)",
            #Engines and APU>>all from camps or the jssi except the programPlan>> from the original or the spec sheet
            """engines: [
              {
                "make": str = Field(description= Manufacturer of the Engine),
                "model": str = Field(description= Model of the Engine),
                "programPlan": str = Field(description= Program Plan of the Engine),
                "serialNumber": str = Field(description= Serial Number of the Engine),
                "totalTimeHours": int = Field(description= Total Flight Time in Hours of the Engine),
                "cycles": int = Field(description= Total Flight Time in Cycles of the Engine),
                "asOfDate": DateTime = Field(description= As Of Date DD/MM/YYYY of the  of the Aircraft or the Plane)
              }
            ]""" ,
            """apu: {
                "make": str = Field(description= "Manufacturer of the APU"),
                "model": str = Field(description= Model of the APU),
                "programPlan": str = Field(description= Program Plan of the APU),
                "serialNumber": str = Field(description= Serial Number of the APU),
                "totalTimeHours": int = Field(description= Total Flight Time in Hours of the APU),
                "cycles": int = Field(description= Total Flight Time in Cycles of the APU,
                "asOfDate": DateTime = Field(description= As Of Date DD/MM/YYYY of the Aircraft or the Plane)
            }""",
            "interior: str = Field(description= interior appearance description of the aircraft or the plane)",
            "exterior: str = Field(description= exterior appearance description of the aircraft or the plane)",
            # """maintenance: {
              # "summary": str = Field(description= MAINTENANCE CHECKS),
              # "scheduledInspections": [
                # {
                  # "title": str = Field(description= Title of the scheduled MAINTENANCE CHECKS.' ),
                  # "performedDate": DateTime = Field(description= Oldest Date DD/MM/YYYY of the MAINTENANCE CHECK),
                  # "performedHours": int = Field(description= Number of performed Hours at the time of the Last MAINTENANCE CHECK),
                  # "dueDate": DateTime = Field(description= Due Date DD/MM/YYYY of the next planed MAINTENANCE CHECK),
                  # "dueHoursOrCycles": int = Field(description= Number of Hours Cycles to the next planed MAINTENANCE CHECK)
                # }
              # ]
            # }""",
            """avionics: [
              str = Field(description= avionics of the Aircraft or the plane)"
            ]""",
            """features: [
              str = Field(description= features of the Aircraft or the plane)"
            ]"""
        ]


 
##########################
    
#+++++++++++++++++++++++++++++++++++++++
    start_time = time.time()
    bm25, model, index,chunks, chunk_map, embedds = extract_context(files,chunk_size=50)
    # for text, (filename, chunk_num), score in results:
        # print(f"\n[File: {filename} | Chunk: {chunk_num} | Score: {score:.4f}]")
        # print(text)

#+++++++++++++++++++++++++++++++++++++++


    
        
    responses_list = []
    combined_dict = {}
    context_buffer = []
    for str_i in required_response:
        which_attribute = str_i.split(":")[0]
        which_attribute = f"***------------------***\nThis is the context of the Json key {which_attribute}"
        # if str_i.startswith("maintenance"):
            # query_text = "\n".join([instructions, f"scheduled_maintenance_logic = {{{scheduled_maintenance_logic}}}", "required_response = {"+str_i+"}",Hard_Rule])
        # elif str_i.startswith("engines"):
            # query_text = "\n".join([instructions, engines, f"required_response = {{{str_i}}}",Hard_Rule])
        # else:
            # query_text = "\n".join([instructions,  "required_response = {"+str_i+"}",Hard_Rule])
        # print("################################## QUERY ####################################")
        # print(query_text)
        
        # if str_i.startswith("maintenance"):
            # results = db.similarity_search_with_score(str_i, k=4)
        # else:
        # results = db.similarity_search_with_score(str_i, k=2)
        #****************************
        # results = keyword_search(str_i, chunks, top_k=1)
        # print("#################################################")
        # print(str_i)
        # print(results)
        # if(results == None):# or (results[0][2]<2):
        results = hybrid_search(str_i, bm25, model, index, chunks, chunk_map, top_k=4, alpha=0)
        result_context = which_attribute+":(\n"+"\n\n---\n\n".join([text for text, (filename, chunk_num), score in results])+"\n)"
        # else:
        #     result_context = which_attribute+":(\n"+"\n\n---\n\n".join([text for text,_, score in results])+"\n)"
        #****************************
        # results = results[::-1]
        
        # result_context = which_attribute+"\n"+"\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        context_buffer.append(result_context)
    end_time = time.time()
    print(f'time Context retrieval {end_time - start_time}')
    # query_text = "\n".join([instructions,engines, f"scheduled_maintenance_logic = {{{scheduled_maintenance_logic}}}", "required_response = {"+"\n".join(required_response)+"}",Hard_Rule])   

    query_text = "\n".join([instructions,engines, "required_response = {"+"\n".join(required_response)+"}",Hard_Rule])   

    # query_text = "\n".join([instructions,engines, "required_response = {"+"\n".join(required_response)+"}",Hard_Rule])       
    
    
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context_buffer])
          
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    context_text = "\n\n".join(context_buffer)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f">>>>>>>>>>number of words in prompt = {len(prompt.strip())}")
    print("################################## CONTEXT ####################################")
    print(context_text)
    
    start_time = time.time()    
    response = generate_response(prompt)#,model="deepseek-r1")
    end_time = time.time()
    print("##############################################")
    print(f'time AI url {end_time - start_time}')
    # print("################################## QUERY ####################################")
    # print(query_text)
    # print(f"NUMBER OF TOKENS IN THE PROMPT = {count_tokens(prompt)}")
    print("################################## RESPONSE ####################################")
    print(response)
    combined_dict.update(extract_json_frm_string(response))
    print(combined_dict)
        
    return combined_dict
    
# query_pdf()
