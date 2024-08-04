"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from parser import langchain_docs_extractor

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from constants import WEAVIATE_DOCS_INDEX_NAME
from langchain_community.document_loaders import RecursiveUrlLoader, DirectoryLoader, TextLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.environ["OPENAI_API_KEY"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


def load_langchain_docs():
    text_loader_kwargs = {'autodetect_encoding': True}
    return DirectoryLoader("./berkeley", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs).load()

def getMetadata(text):
    prompt = f"""
    Extract the property type(house/apartment), price (in £), nr bedrooms, and internal square feet from the following 
    property. Return comma-separated string in the format: [ptype], [price], [beds], [feet]. Determine single
    value for integers always, you decide, its no problem if it is wrong.
    Property Description:
    {text}
    Example output:
    apartment, 100000, 3, 1393, https://admin/Avenue-13_Lo.jpg
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    message_content = response.choices[0].message.content.strip()

     # Ensure message content is not empty and parse the result
    if not message_content:
        return {"ptype": "apartment", "price": 1500000, "beds": 1, "feet": 100}
    try:
        ptype, price, beds, feet = message_content.split(', ', 4)
        price = int(price.replace('£', '').replace(',', '').strip())
        beds = int(beds.strip())
        feet = int(feet.replace(',', '').strip())
    except (ValueError, IndexError) as e:
        print(f'Error parsing: {e}')
        return {"ptype": "apartment", "price": 1500000, "beds": 1, "feet": 100}

    returnValue = {"ptype":ptype, "price":price, "beds": beds, "feet": feet}
    return returnValue

def vectorTest(docs_transformed, record_manager, vectorstore):
    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Vector Database Test: Indexing stats: {indexing_stats}")

def ingest_docs():
    
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "ptype", "price", "beds", "feet"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    #Test connection before fetching docs (and metadata extraction from llm)
    vectorTest([], record_manager, vectorstore)

    docs_from_documentation = load_langchain_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")

        # retrieved document.
    for doc in docs_from_documentation:
        extractedMetadata = getMetadata(doc.page_content)
        txtSource = doc.metadata["source"]
        if extractedMetadata :
            doc.metadata["source"] = doc.page_content.split(',')[0].replace('Property_link: ', '')
            doc.metadata["ptype"]=extractedMetadata["ptype"]
            doc.metadata["price"]=extractedMetadata["price"]
            doc.metadata["beds"]=extractedMetadata["beds"]
            doc.metadata["feet"]=extractedMetadata["feet"]
            # Extract image URLs using a loop
            image_urls = []
            for part in doc.page_content.split(','):
                match = re.search(r'Image: (.+)', part)
                if match:
                    image_urls.append(match.group(1).strip())  # Extract the URL part and strip any extra spaces
            #Limit to last 2 images only. Otherwise it gets too much on frontend
            doc.metadata["images"] = ', '.join(image_urls[-2:])
            print(f"Extracted metadata {doc.metadata}")

    docs_transformed = text_splitter.split_documents(
        docs_from_documentation
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )


if __name__ == "__main__":
    ingest_docs()
