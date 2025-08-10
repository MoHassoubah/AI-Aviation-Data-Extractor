
import pdfplumber
import os
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import random


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

def split_into_chunks(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def create_chunks(documents,chunk_size, filenames):
    chunks = []
    chunk_map = []

    for i, doc in enumerate(documents):
        doc_chunks = split_into_chunks(doc,chunk_size)
        
        chunks.extend(doc_chunks)
        chunk_map.extend([(filenames[i], j) for j in range(len(doc_chunks))])

    return chunks, chunk_map


required_response = [
#   "make: Embraer",
#   "model: EMB-135BJ Legacy 650",
#   "serialNumber: 14501198",
#   "registration: N1977H",
#   "manufacturer: Embraer",
#   "airframeProgram: Embraer Executive Care Enhanced",
#   "totalTimeHours 2079",
#   "totalCycles 1204",
#   "flightHoursAsOfDate 10-DEC-2022",
#   "entryToServiceDate 26-FEB-2014",
#   """
#   "engines": [
#     {
#       "make": "Rolls Royce DERBY Plc",
#       "model": "AE3007A2",
#       "programPlan": "Rolls Royce Corporate Care",
#       "serialNumber": "CAE313365",
#       "totalTimeHours": 2104,
#       "cycles": 1204,
#       "asOfDate": "10-DEC-2022"
#     },
#     {
#       "make": "Rolls Royce DERBY Plc",
#       "model": "AE3007A2",
#       "programPlan": "Rolls Royce Corporate Care",
#       "serialNumber": "CAE313364",
#       "totalTimeHours": 2104,
#       "cycles": 1204,
#       "asOfDate": "10-DEC-2022"
#     }
#   ]""",
#   """
#   "apu": {
#     "make": "Pratt & Whitney - Hamilton Standard – Sundstrand",
#     "model": "AP5500R/T-62T-40C14",
#     "programPlan": "Embraer Executive Care Enhanced",
#     "serialNumber": "SP-E1328158",
#     "totalTimeHours": 1802,
#     "cycles": 1204,
#     "asOfDate": "10-DEC-2022"
#   }""",
#   "interior: 13 Passengers + Crew + Observer Seat; Forward Galley; Dual Lavatories; Club Arrangement; 4-Place Dining; 3-Place Divan; Dual Blu-Ray; Airshow 4000; Dual 19-Inch & Side Ledge Monitors; Ovation Cabin Management System; GoGo Biz L-5 Wi-Fi",
  "exterior: White with Silver and Blue",
  """
  "maintenance": {
    "summary": "Embraer Low Utilization Plan with CAMP Systems Computerized Aircraft Maintenance Tracking Program",
    "scheduledInspections": {
      "012M_0500H": {
        "performedDate": "10-May-2022",
        "performedHours": 1665,
        "dueDate": "10-May-2023",
        "dueHoursOrCycles": 2165
      },
      "024M_1000H": {
        "performedDate": "01-May-2022",
        "performedHours": 1665,
        "dueDate": "01-May-2024",
        "dueHoursOrCycles": 2665
      },
      "048M_2000H": {
        "performedDate": "01-May-2022",
        "performedHours": 1665,
        "dueDate": "01-May-2026",
        "dueHoursOrCycles": 3665
      },
      "072M_3000H": {
        "performedDate": "01-FEB-2020",
        "performedHours": 1463,
        "dueDate": "01-FEB-2026",
        "dueHoursOrCycles": 4463
      },
      "096M_4000H": {
        "performedDate": "01-May-2022",
        "performedHours": 1665,
        "dueDate": "01-May-2030",
        "dueHoursOrCycles": 5665
      },
      "144M_6000H": {
        "performedDate": null,
        "performedHours": null,
        "dueDate": "01-FEB-2026",
        "dueHoursOrCycles": 6000
      },
      "192M_8000H": {
        "performedDate": null,
        "performedHours": null,
        "dueDate": "01-FEB-2030",
        "dueHoursOrCycles": 8000
      },
      "288M_12000H": {
        "performedDate": null,
        "performedHours": null,
        "dueDate": "01-FEB-2038",
        "dueHoursOrCycles": 12000
      },
      "Gear_144M_Overhaul": {
        "performedDate": null,
        "performedHours": null,
        "dueDate": "01-FEB-2026",
        "dueHoursOrCycles": null
      }
    }
  }""",
  "avionics: Honeywell RNZ-851 NAV/COMM/ADF/DME; RCZ-833K VHF Comms; Artex C406-2 ELT; Jet Call SELCAL; CM-950 Cabin Management; WX-880 Weather Radar; KRX-1053 HF Comm; MARK V EGPWS; NZ-2000 FMS 6.1; RT-300 Radio Altimeter; GR-550 GPS; TCAS 2000 Change 7.1; IC-600 Integrated Computers; Laseref IV IRS",
  "features: FANS1/A; CPDLC; ADS-B Out V2; TCAS 7.1; WAAS; LPV; RVSM; RNP 0.3; Iridium Phone; GoGo Biz L-5 High Speed Data and Wi-Fi"
 ]

prompts = [
            # make, model, serialNumber, registration, manufacturer>> from all documents
            # "make: str = Field(description= Manufacturer name of the aircraft or the plane)",
            # "model: str = Field(description= Model name of the aircraft or the plane)",
            # "serialNumber: str = Field(description= s/n Serial Number specific to this aircraft or the plane. It's only number no letters.)",
            # "registration: str = Field(description= Registration number of the aircraft or the plane. It comes after the serial number by |)",
            # "manufacturer: str = Field(description= Manufacturer name of the aircraft or the plane)",
            #mainly from the spec sheet
            # "airframeProgram:  str = Field(description= Airframe Power-by-The-Hour Plan of the aircraft or the plane)",
            #totalTimeHours,totalCycles>> all documents (take the largest number)
            # "totalTimeHours: int = Field(description= total Flight Time in Hours of the aircraft or the plane. If there are multiple correspoinding values,take the largest value.)",
            # "totalCycles: int = Field(description= total Flight Time in Cycles of the aircraft or the plane. If there are multiple correspoinding values,take the largest value.)",
            #flightHoursAsOfDate, entryToServiceDate>> from camps and jssi, (take the lastest date for flightHoursAsOfDate)
            # "flightHoursAsOfDate: DateTime = Field(description= flight Hours As Of Date DD/MM/YYYY of the aircraft or the plane)",
            # "entryToServiceDate: DateTime = Field(description= Date DD/MM/YYYY of the First Service of the aircraft or the plane)",
            #Engines and APU>>all from camps or the jssi except the programPlan>> from the original or the spec sheet
            # """engines: [
            #   {
            #     "make": str = Field(description= Manufacturer name of the Engine),
            #     "model": str = Field(description= Model name of the Engine),
            #     "programPlan": str = Field(description= Program Plan of the Engine),
            #     "serialNumber": str = Field(description= Serial Number of the Engine),
            #     "totalTimeHours": int = Field(description= Total Flight Time in Hours of the Engine),
            #     "cycles": int = Field(description= Total Flight Time in Cycles of the Engine),
            #     "asOfDate": DateTime = Field(description= As Of Date DD/MM/YYYY of the  of the Aircraft or the Plane)
            #   }
            # ]""" ,
            # """apu: {
            #     "make": str = Field(description= "Manufacturer name of the APU"),
            #     "model": str = Field(description= Model name of the APU),
            #     "programPlan": str = Field(description= Program Plan of the APU),
            #     "serialNumber": str = Field(description= Serial Number of the APU),
            #     "totalTimeHours": int = Field(description= Total Flight Time in Hours of the APU),
            #     "cycles": int = Field(description= Total Flight Time in Cycles of the APU,
            #     "asOfDate": DateTime = Field(description= As Of Date DD/MM/YYYY of the Aircraft or the Plane)
            # }""",
            # "interior: str = Field(description= interior appearance description of the aircraft or the plane)",
            "exterior: str = Field(description= exterior appearance description of the aircraft or the plane)",
            """maintenance: {
              "summary": str = Field(description= MAINTENANCE CHECKS),
              "scheduledInspections": [
                {
                  "title": str = Field(description= Title of the scheduled MAINTENANCE CHECKS.' ),
                  "performedDate": DateTime = Field(description= Oldest Date DD/MM/YYYY of the MAINTENANCE CHECK),
                  "performedHours": int = Field(description= Number of performed Hours at the time of the Last MAINTENANCE CHECK),
                  "dueDate": DateTime = Field(description= Due Date DD/MM/YYYY of the next planed MAINTENANCE CHECK),
                  "dueHoursOrCycles": int = Field(description= Number of Hours Cycles to the next planed MAINTENANCE CHECK)
                }
              ]
            }""",
            """avionics: [
              str = Field(description= avionics of the Aircraft or the plane)"
            ]""",
            """features: [
              str = Field(description= features of the Aircraft or the plane)"
            ]"""
        ]

def create_dataframes_query_pos_neg_examples():
    folder_path =  ".././data_mail_extract"
    documents = []
    filenames = []
    data = []
    chunk_check_window = 10

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".pdf"):
            print(file_path)
            text = extract_text_table_from_pdf(file_path)
            documents.append(text.lower())
            filenames.append(filename)

    chunks, chunk_map = create_chunks(documents, chunk_size=50, filenames=filenames)
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    for prompt_index, query in enumerate(required_response):
        bm25_scores = bm25.get_scores(query.lower().split())
        decen_indices = np.argsort(bm25_scores)[::-1]
        chunks_curr = np.array(chunks)[decen_indices]
        bm25_scores = bm25_scores[decen_indices]
        print(query)
        print(prompts[prompt_index])
        print("top cands with scores")
        # print(bm25_scores[decen_indices][:chunk_check_window])
        # print(np.array(chunks)[decen_indices][:chunk_check_window])
        pos_match_fnd = False
        for i in range(0,len(chunks_curr),chunk_check_window):
            for j in range(chunk_check_window):
                print(f"\nindex ={i+j}   score = {bm25_scores[i+j]}   {chunks_curr[i+j]}")

            ids_str = input("what index one or more?")
            ids_lst = [int(x) for x in ids_str.split(',') if x.strip() != '']
            if len(ids_lst) > 0:
                pos_match_fnd = True
                for index in ids_lst:
                    dataframe_rw = pd.DataFrame([{"query": prompts[prompt_index], "pos_context": chunks_curr[index], "neg_context": chunks_curr[-1]}])
                    print(f'chosen data frame row>> {dataframe_rw}')
                    # data.append(dataframe_rw)
                    write_header = not os.path.exists('./embedding_dataset.csv')

                    dataframe_rw.to_csv('./embedding_dataset.csv', mode='a', header=write_header, index=False)
                break
            print("*********************************************")
            if pos_match_fnd:
                break
    
    # df = pd.DataFrame(data)
    # df.to_csv("embedding_dataset.csv", index=False)

# create_dataframes_query_pos_neg_examples()


def eval_finetuned_embedd_models(checkpoint_cnt):
    folder_path =  ".././data_mail_extract"
    documents = []
    filenames = []
    data = []
    chunk_check_window = 10

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".pdf"):
            print(file_path)
            text = extract_text_table_from_pdf(file_path)
            documents.append(text.lower())
            filenames.append(filename)

    chunks, chunk_map = create_chunks(documents, chunk_size=50, filenames=filenames)
    
    model_trianed = SentenceTransformer('./models/aviation-finetuned-all-MiniLM-L6-v2/checkpoint-'+str(checkpoint_cnt))
    embeddings_trianed = model_trianed.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings_trianed)
    index_trained = faiss.IndexFlatIP(embeddings_trianed.shape[1])  # Inner Product for cosine
    index_trained.add(embeddings_trianed)
    
    model_init = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_init = model_init.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings_init)
    index_init = faiss.IndexFlatIP(embeddings_init.shape[1])  # Inner Product for cosine
    index_init.add(embeddings_init)

    end_point = 0

    df = pd.read_csv("embedding_dataset.csv")
    with open('./logs/output'+str(checkpoint_cnt)+'.txt', "w", encoding="utf-8") as f:
        for index, row in df.iterrows():
            query = row['query']
            pos = row['pos_context']
            neg = row['neg_context']

            if "make: str = Field(description= Manufacturer name of the aircraft or the plane)" in query:
                end_point = end_point+1

            f.write(f"query is >>{query}\n\n")
            print(f"query is >>{query}")
            print()
            f.write(f"pos is >>{pos}\n\n")
            print(f"pos is >>{pos}")
            print()

            query_vec_trianed = model_trianed.encode([query], convert_to_numpy=True)
            # query_vec = np.array(model.embed_documents([query]), dtype='float32')
            faiss.normalize_L2(query_vec_trianed)
            D_trianed, I_trianed = index_trained.search(query_vec_trianed, len(chunks))
            faiss_scores_trianed = np.zeros(len(chunks))
            faiss_scores_trianed[I_trianed[0]] = D_trianed[0]
            top2bottom_indices_trianed = np.argsort(faiss_scores_trianed)[::-1]
            # chunks_ordered_trained = chunks[top2bottom_indices_trianed]
            if pos in chunks:
                fnd_index_trained  = chunks.index(pos)
                print("score:", faiss_scores_trianed[fnd_index_trained])
                print("Found at index_trained:", list(top2bottom_indices_trianed).index(fnd_index_trained))
                f.write(f"score: {faiss_scores_trianed[fnd_index_trained]}\n")
                f.write(f"Found at index_trained: {list(top2bottom_indices_trianed).index(fnd_index_trained)}\n\n")
            else:
                print("index_trained Not found")
                f.write("index_trained Not found\n\n")


            print()
            
            query_vec_init = model_init.encode([query], convert_to_numpy=True)
            # query_vec = np.array(model.embed_documents([query]), dtype='float32')
            faiss.normalize_L2(query_vec_init)
            D_init, I_init = index_init.search(query_vec_init, len(chunks))
            faiss_scores_init = np.zeros(len(chunks))
            faiss_scores_init[I_init[0]] = D_init[0]
            top2bottom_indices_init = np.argsort(faiss_scores_init)[::-1]
            # chunks_ordered_init = chunks[top2bottom_indices_init]
            
            if pos in chunks:
                fnd_index_init  = chunks.index(pos)
                print("score:", faiss_scores_init[fnd_index_init])
                print("Found at index_init:", list(top2bottom_indices_init).index(fnd_index_init))
                f.write(f"score: {faiss_scores_init[fnd_index_init]}\n")
                f.write(f"Found at index_init: {list(top2bottom_indices_init).index(fnd_index_init)}\n\n")
            else:
                print("index_init Not found")
                f.write("index_trained Not found\n\n")

            # input()
            if end_point > 3:
                break


def prepare_new_dataset_after_1st_finetuning():
    folder_path =  ".././data_mail_extract"
    documents = []
    filenames = []
    data = []
    chunk_check_window = 10

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".pdf"):
            print(file_path)
            text = extract_text_table_from_pdf(file_path)
            documents.append(text.lower())
            filenames.append(filename)

    chunks, chunk_map = create_chunks(documents, chunk_size=50, filenames=filenames)
    
    model_trianed = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_trianed = model_trianed.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings_trianed)
    index_trained = faiss.IndexFlatIP(embeddings_trianed.shape[1])  # Inner Product for cosine
    index_trained.add(embeddings_trianed)
    
    df = pd.read_csv("embedding_dataset.csv")

    
    pos_map = {}
    neg_map = {}
    for _, row in df.iterrows():
        query = row["query"]
        pos_map.setdefault(query, []).append(row["pos_context"])
        neg_map.setdefault(query, []).append(row["neg_context"])

    for query, pos_lst in pos_map.items():
        query_vec_trianed = model_trianed.encode([query], convert_to_numpy=True)
        # query_vec = np.array(model.embed_documents([query]), dtype='float32')
        faiss.normalize_L2(query_vec_trianed)
        D_trianed, I_trianed = index_trained.search(query_vec_trianed, len(chunks))
        faiss_scores_trianed = np.zeros(len(chunks))
        faiss_scores_trianed[I_trianed[0]] = D_trianed[0]
        top2bottom_indices_trianed = np.argsort(faiss_scores_trianed)[::-1]
        # chunks_ordered_trained = chunks[top2bottom_indices_trianed]
        pos_rank_list=[]
        for pos in pos_lst:
            if pos in chunks:
                fnd_index_trained  = chunks.index(pos)
                print("score:", faiss_scores_trianed[fnd_index_trained])
                pos_rank = list(top2bottom_indices_trianed).index(fnd_index_trained)
                print("Found at index_trained:", pos_rank)
                pos_rank_list.append(pos_rank)
            else:
                print("index_trained Not found")
        pos_w_highest_score = pos_lst[pos_rank_list.index(min(pos_rank_list))]
        rank_pos_w_lowest_score = max(pos_rank_list)
        if rank_pos_w_lowest_score>4:
            for i_cnt in range(rank_pos_w_lowest_score):
                chunk_order_index = top2bottom_indices_trianed[i_cnt]
                neg_chunk = chunks[chunk_order_index]
                if neg_chunk in pos_lst:
                    continue
                dataframe_rw = pd.DataFrame([{"query": query, "pos_context": random.choice(pos_lst), "neg_context": neg_chunk}])
                # print(f'chosen data frame row>> {dataframe_rw}')
                # data.append(dataframe_rw)
                write_header = not os.path.exists('./embedding_dataset.csv')

                dataframe_rw.to_csv('./embedding_dataset.csv', mode='a', header=write_header, index=False)
        # for pos_chunk, pos_rank in zip(pos_lst, pos_rank_list):



# prepare_new_dataset_after_1st_finetuning()
iter_to_check_lst = [500]
for chk_pnt in iter_to_check_lst:
    eval_finetuned_embedd_models(chk_pnt)



