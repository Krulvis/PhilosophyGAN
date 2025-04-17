from langchain_ollama import OllamaEmbeddings

class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """

    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function with Ollama embeddings

embedder = OllamaEmbeddings(
    model="llama3",
    base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
)
embedding_func = ChromaDBEmbeddingFunction(embedder)