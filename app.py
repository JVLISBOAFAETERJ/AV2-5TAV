import os, pdb
import re
import requests
import streamlit as st
import json
from pprint import pprint
from pathlib import Path
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain_community.document_loaders import PyMuPDFLoader, JSONLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.agents import AgentType, initialize_agent
from langchain_community.document_loaders import TextLoader
from htmlTemplates import user_template, bot_template, css

set_debug(True)

class CNPJTool:
    def __init__(self):
        self.cnpj_pattern = re.compile(r'\d{2}\.\d{3}\.\d{3}\/\d{4}\-\d{2}')
        
    def extract_cnpjs(self, text):
        """
        Extracts CNPJs from the given text.
        
        Args:
            text (str): The input text from which to extract CNPJs.
        
        Returns:
            list: A list of extracted CNPJs.
        """
        return self.cnpj_pattern.findall(text)
    
    def fetch_cnpj_info(self, cnpj):
        """
        Fetches information for the given CNPJ from the Receita WS API.
        
        Args:
            cnpj (str): The CNPJ number in the format XX.XXX.XXX/XXXX-XX.
        
        Returns:
            dict: The JSON response from the API as a dictionary, or None if the request fails.
        """
        api_url = f"https://www.receitaws.com.br/v1/cnpj/{cnpj.replace('.', '').replace('/', '').replace('-', '')}"
        response = requests.get(api_url)
        if response.status_code == 200:
            response_json = response.json() 
            if response_json["status"] != "ERROR":
                response_json_formated = self.format_cnpj_info(response_json)
                return response_json_formated
        else:
            return None
        
    def format_cnpj_info(self, info):
        formatted_info = []

        formatted_info.append(f'A empresa "{info["nome"]}" foi aberta em {info["abertura"]} e está atualmente em situação {info["situacao"].lower()} como uma {info["tipo"].lower()}.')
        formatted_info.append(f'Seu nome fantasia é {"desconhecido" if not info["fantasia"] else info["fantasia"]}.')
        formatted_info.append(f'A natureza jurídica da empresa é "{info["natureza_juridica"]}" e seu porte é classificado como "{info["porte"]}".')
        formatted_info.append(f'A atividade principal da empresa é "{info["atividade_principal"][0]["text"]}" (código {info["atividade_principal"][0]["code"]}).')

        if info["atividades_secundarias"]:
            formatted_info.append('Além da atividade principal, a empresa também exerce atividades secundárias, como:')
            for atividade in info["atividades_secundarias"]:
                formatted_info.append(f'"{atividade["text"]}" (código {atividade["code"]}).')

        if info["qsa"]:
            formatted_info.append('O quadro societário da empresa inclui os seguintes membros:')
            for qsa in info["qsa"]:
                formatted_info.append(f'{qsa["nome"]} ({qsa["qual"]})')

        formatted_info.append(f'A empresa está localizada na {info["logradouro"]}, número {info["numero"]}, {info["complemento"]}, no bairro {info["bairro"]}, município de {info["municipio"]}, {info["uf"]}, CEP {info["cep"]}.')
        formatted_info.append(f'O telefone para contato é {info["telefone"]}.')
        formatted_info.append(f'O email para contato é {info["email"]}.')
        formatted_info.append(f'A situação da empresa foi estabelecida em {info["data_situacao"]}, e o CNPJ é {info["cnpj"]}.')
        formatted_info.append(f'A última atualização dos dados foi em {info["ultima_atualizacao"]}.')
        formatted_info.append(f'O status da empresa é {info["status"]}.')
        formatted_info.append(f'O capital social da empresa é de R$ {float(info["capital_social"]):,.2f}.')
        formatted_info.append(f'Fim dos dados desse CNPJ.')

        return formatted_info

# Função para carregar o documento PDF
def load_pdf_doc(pdf_file):
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / pdf_file.name
    with open(temp_file, mode="wb") as f:
        f.write(pdf_file.getvalue())

    doc = PyMuPDFLoader(temp_file).load()

    if isinstance(temp_file, Path):
        os.remove(temp_file)
        Path.rmdir(temp_dir)

    return doc

def load_text_doc(text_lines):
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)
    txtname = "dados_receita_federal.txt"
    temp_file = temp_dir / txtname
    print(text_lines)
    with open(temp_file, mode="w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    loader = TextLoader(temp_file, encoding="utf-8")
    doc = loader.load()
    print(doc)
    if isinstance(temp_file, Path):
        os.remove(temp_file)
        temp_dir.rmdir()

    return doc

# Função para carregar o documento json
def load_json_doc(json_file):
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)
    jsonname = json_file["Dados da Empresa"]["CNPJ"].replace('.', '').replace('/', '').replace('-', '') + ".json"
    temp_file = temp_dir / jsonname
    json.dump(json_file, open(temp_file, "w", encoding="utf-8", ensure_ascii=False))

    loader = JSONLoader(
    file_path=temp_file,
    jq_schema='.',
    text_content=False)

    doc = loader.load()
    
    if isinstance(temp_file, Path):
        os.remove(temp_file)
        Path.rmdir(temp_dir)

    return doc

# Função para carregar o documento excel
def load_excel_doc(excel_file):
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / excel_file.name
    with open(temp_file, mode="wb") as f:
        f.write(excel_file.getvalue())

    loader = UnstructuredExcelLoader(temp_file, mode='elements')
    doc = loader.load()

    if isinstance(temp_file, Path):
        os.remove(temp_file)
        Path.rmdir(temp_dir)

    return doc

def load_file(file):
    if file.name.endswith('.pdf'):
        return load_pdf_doc(file)

    if file.name.endswith('.xlsx'):
        return load_excel_doc(file)

# Função para obter os chunks de texto do documento
def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    chunks = text_splitter.split_documents(docs)
            
    return chunks

# Função para obter a store de vetores a partir dos chunks de texto
def get_vectorstore(text_chunks):
    embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)

    return vectorstore

def get_conversation_agent(llm, tools):
    llm = AzureChatOpenAI(
        api_version="2023-12-01-preview", azure_deployment="gpt-4-turbo-2024-04-09"
    )
    
    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
    )

    return agent

# Função para lidar com a entrada do usuário
def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({"input": user_question})
    st.write(
        user_template.replace("{{MSG}}", response["input"]), unsafe_allow_html=True
    )
    st.write(
        bot_template.replace("{{MSG}}", response["output"]), unsafe_allow_html=True
    )

# Função principal
def main():
    load_dotenv()
    st.set_page_config(page_title="Assistente Jurídico", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Assistente Jurídico")
    user_question = st.text_input("Faça uma pergunta sobre seus documentos:")

    if user_question:
        handle_userinput(user_question)

    llm = AzureChatOpenAI(
        api_version="2023-12-01-preview", azure_deployment="gpt-4-turbo-2024-04-09"
    )

    with st.sidebar:
        st.subheader("Seus documentos")
        files = st.file_uploader("Envie seus documentos", accept_multiple_files=True)
        tools = []

        if st.button("Process"):
            with st.spinner("Processando"):
                docs = []
                cnpj_tool = CNPJTool()
                all_cnpjs = set()

                # Processar cada arquivo PDF
                for file in files:
                    doc = load_file(file)
                    docs.append(doc)
                    full_text = "".join([page.page_content for page in doc])
                    cnpjs = cnpj_tool.extract_cnpjs(full_text)
                    all_cnpjs.update(cnpjs)

                    # Criar vectorstore para o documento individual
                    text_chunks = get_text_chunks(doc)
                    vectorstore = get_vectorstore(text_chunks)
                    retriever = vectorstore.as_retriever()
                    tools.append(
                        Tool(
                            name=re.sub('[^a-zA-Z0-9_-]', '', file.name),
                            description=f"useful when you want to answer questions about {file.name}",
                            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
                        )
                    )

                # Processar informações dos CNPJs
                cnpj_info_docs = []
                for cnpj in all_cnpjs:
                    info = cnpj_tool.fetch_cnpj_info(cnpj)
                    if info:
                        cnpj_info_docs.append(load_text_doc(info)[0])
                print(cnpj_info_docs)
                if cnpj_info_docs:
                    # Criar vectorstore para o documento de informações dos CNPJs
                    cnpj_info_chunks = get_text_chunks(cnpj_info_docs)
                    cnpj_info_vectorstore = get_vectorstore(cnpj_info_chunks)
                    cnpj_info_retriever = cnpj_info_vectorstore.as_retriever()
                    tools.append(
                        Tool(
                            name="CNPJ_receita_federal_database",
                            description="useful when you want to answer questions about the CNPJ Receita Federal database",
                            func=RetrievalQA.from_chain_type(llm=llm, retriever=cnpj_info_retriever),
                        )
                    )

                st.session_state.conversation = get_conversation_agent(llm, tools)

if __name__ == "__main__":
    main()
