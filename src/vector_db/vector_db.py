from typing import List

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class VectorDB:

    def __init__(self, documents: List[Document], k: int, embed_model: HuggingFaceEmbedding, splitter: NodeParser,
                 verbose: bool):
        self.splitter = splitter
        self.embed_model = embed_model
        self.n_chunks = None
        self._documents = documents
        self._k = k
        self.verbose = verbose
        self.vector_index = self.split_documents()
        self.retriever = self.create_retriever()

    def split_documents(self) -> VectorStoreIndex:
        vector_index = VectorStoreIndex.from_documents(documents=self._documents, show_progress=self.verbose,
                                                       transformations=[self.splitter], embed_model=self.embed_model)
        return vector_index

    def create_retriever(self):
        retriever = self.vector_index.as_retriever(similarity_top_k=self._k, embed_model=self.embed_model)
        return retriever

    def retrieve(self, query: str) -> List[NodeWithScore]:
        return self.retriever.retrieve(str_or_query_bundle=query)
