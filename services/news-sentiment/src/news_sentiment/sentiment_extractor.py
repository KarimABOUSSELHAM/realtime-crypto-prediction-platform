import os
from typing import Literal, Optional

from baml_py import ClientRegistry
from loguru import logger
from opik import track

from news_sentiment.baml_client.sync_client import b
from news_sentiment.baml_client.types import SentimentScores


class SentimentExctractor:
    def __init__(
        self,
        model,
        base_url: Optional[str] = 'http://localhost:11434/v1',  # Ollama default
    ):
        self.model = model
        self.base_url = base_url
        model_provider, model_name = model.split('/')
        logger.debug(
            f'Initializing SentimentExctractor with model: {model_provider}/{model_name}'
        )
        self._client_registry = self._init_client_registry(model_provider, model_name)

    def _init_client_registry(
        self, model_provider: Literal['anthropic', 'openai-generic'], model_name: str
    ) -> ClientRegistry:
        """
        Initialize the client registry for the sentiment extractor model.
        This method should be implemented to return the appropriate client registry.
        """
        # Placeholder for actual implementation
        cr = ClientRegistry()
        if model_provider == 'anthropic':
            # Creates a new client for Anthropic models
            cr.add_llm_client(
                name='MyDynamicClient',
                provider='anthropic',
                options={
                    'model': model_name,
                    'temperature': 0.0,
                    'api_key': os.environ.get('ANTHROPIC_API_KEY'),
                },
            )
        elif (
            model_provider == 'openai-generic'
        ):  # `openai-generic` stands for local ollama models
            cr.add_llm_client(
                name='MyDynamicClient',
                provider='openai-generic',
                options={
                    'model': model_name,
                    'temperature': 0.0,
                    'base_url': self.base_url,
                },
            )
        else:
            raise ValueError(f'Unsupported model provider: {model_provider}')
        # Sets MyAmazingClient as the primary client
        cr.set_primary('MyDynamicClient')
        return cr

    @track
    def extract_sentiment_scores(self, news: str) -> SentimentScores:
        """
        Extract the sentiment scores for the given news

        Args:
            news (str): news given to score

        Returns:
            SentimentScores: Object instance which includes the scores
        """
        return b.ExtractSentimentScores(
            news,
            {'client_registry': self._client_registry},
        )


if __name__ == '__main__':
    sentiment_extractor = SentimentExctractor(model='ClaudeOpus4')
    print(
        sentiment_extractor.extract_sentiment_scores(
            'Goldman Sachs says it is going to buy soon 1 M BTC and sell 1 M ETH.'
        )
    )
