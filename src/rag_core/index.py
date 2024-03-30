import qdrant_client
from typing import List
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import Document,TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.config import AppConfig
import os

loaders={'.pdf': PyMuPDFReader()}

settings=AppConfig()

class IndexData():
    """Index the given data into qdrant vectorstore"""

    def __init__(
        self,
        collection_name:str,
        embed_model_name:str="BAAI/bge-small-en",
        folder_path:str="./data/",
        supported_doc_types: List[str] = ['.pdf']
    ) -> None:
        
        """Init params."""
        self.collection_name=collection_name
        self.folder_path = folder_path
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        self.supported_types=supported_doc_types


    def load_data(self):
        files = os.listdir(self.folder_path)
        all_files = list(map(lambda name: os.path.join(self.folder_path, name), files))

        assert all(map(lambda x: x.endswith('.pdf'),all_files))
        
        all_documents=[]
        for file in all_files:
            extension = os.path.splitext(file)[1]
            documents = loaders[extension].load(file_path=file)
            all_documents.append(documents)

        return all_documents
    
    def prepare_data(self,documents:List[Document]):

        text_parser = SentenceSplitter(
                chunk_size=1024,
                # separator=" ",
            )

        text_chunks = []
        # maintain relationship with source doc index, to help inject doc metadata in (3)
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)


        for node in nodes:
            node_embedding = self.embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        return nodes


    def index_data(self):

        client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
        vector_store = QdrantVectorStore(
            client=client, collection_name=self.collection_name
)

        documents=self.load_data()
        all_nodes=[]

        for document in documents:
            doc_nodes=self.prepare_data(documents=document)
            all_nodes.extend(doc_nodes)

        return vector_store.add(all_nodes)