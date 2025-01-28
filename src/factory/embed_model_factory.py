from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config.config import EmbedModelConfig


class EmbedModelFactory:

    @staticmethod
    def create(embed_model_config: EmbedModelConfig):
        return HuggingFaceEmbedding(
            model_name=embed_model_config.embed_model_name,
            device=embed_model_config.embed_model_device,
            trust_remote_code=True
        )
