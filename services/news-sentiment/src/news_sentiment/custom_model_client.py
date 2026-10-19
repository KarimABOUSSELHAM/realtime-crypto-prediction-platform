# Building custom ollama model client and using it

import uuid

import ollama
import requests
from adalflow.core.model_client import ModelClient
from adalflow.core.types import EmbedderOutput, GeneratorOutput, ModelType
from adalflow.optim.parameter import Parameter
from openai.types import (
    CreateEmbeddingResponse,
)


class OllamaModelClient(ModelClient):
    def __init__(self, ollama_client=None, model_name='deepseek-r1:8b'):
        # Call the parent class's initializer
        super().__init__()
        self.ollama_client = ollama_client
        self.model_name = model_name

    # Method to convert input into API parameters for different model types (LLM or Embedder)
    def convert_inputs_to_api_kwargs(
        self, input=None, model_kwargs=None, model_type=ModelType.UNDEFINED
    ):
        model_kwargs = model_kwargs or {}
        if model_type == ModelType.LLM:
            return {
                'model': model_kwargs['model'],
                'messages': input,
                'temperature': model_kwargs.get('temperature', 1),
                'max_tokens': model_kwargs.get('max_tokens', 1024),
            }
        elif model_type == ModelType.EMBEDDER:
            return {
                'model': model_kwargs['model'],
                'input': [input],
            }
        else:
            raise ValueError(f'model_type {model_type} is not supported')

    # Method to make the actual API call to Ollama for either completions (LLM) or embeddings
    def call(self, api_kwargs=None, model_type=ModelType.UNDEFINED):
        """
        Call the appropriate Ollama method based on the model type (LLM or Embedder).

        Args:
            api_kwargs (dict): Arguments to be passed to the API call.
            model_type (ModelType): The type of model (LLM or Embedder).

        Returns:
            Response: The API response from Ollama.
        """
        api_kwargs = api_kwargs or {}
        if model_type == ModelType.LLM:
            api_url = 'http://localhost:11435/v1/chat/completions'
            # payload = {
            #     "model": api_kwargs.get("model"),
            #     "messages": api_kwargs.get("messages"),
            #     "temperature": api_kwargs.get("temperature", 1),
            #     "max_tokens": api_kwargs.get("max_tokens", 1024),
            # }
            payload = {
                'model': api_kwargs.get('model'),
                'messages': [{'role': 'user', 'content': api_kwargs.get('messages')}],
                'temperature': api_kwargs.get('temperature', 1),
                'max_tokens': api_kwargs.get('max_tokens', 1024),
            }
            # print("Ollama payload:", payload)

            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            content = response.json()

            # Create a Parameter with data_id
            param = Parameter(name='response', data=content, data_id=str(uuid.uuid4()))

            # Wrap in GeneratorOutput
            return GeneratorOutput(data=param)

        elif model_type == ModelType.EMBEDDER:
            # Ollama does not have an embedder API, raise if called
            raise NotImplementedError('Ollama embedder is not supported yet.')

        else:
            raise ValueError(f'Unsupported model type: {model_type}')

    # Method to parse the response from a chat completion call
    def parse_chat_completion(self, completion):
        """
        Parse the response from a chat completion into a custom output format.

        Args:
            completion: The completion response from Ollama.

        Returns:
            GeneratorOutput: A custom data structure containing the parsed response.
        """
        return GeneratorOutput(
            data=completion,  # Store the raw completion data
            error=None,  # No error in this case
            raw_response=str(completion),  # Store the raw response as a string
        )

    # Method to parse the response from an embedding API call
    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        """
        Parse the response from an embedding API call into a custom output format.

        Args:
            response (CreateEmbeddingResponse): The response from the embedding API.

        Returns:
            EmbedderOutput: A custom data structure containing the parsed embedding response.
        """
        pass


def build_custom_model_client():
    # Instantiate the custom model client (SimpleCustomModelClient)
    custom_client = OllamaModelClient(ollama_client=ollama, model_name='deepseek-r1:8b')

    # Define the query for the model to process
    query = 'What is the capital of France?'

    # Set the model type for a Large Language Model (LLM)
    model_type = ModelType.LLM

    # Prepare the message prompt as expected by the OpenAI chat API.
    # This format is suitable for GPT-like models (e.g., gpt-3.5-turbo).
    message_prompt = [
        {
            'role': 'user',  # Define the user role in the conversation
            'content': [
                {
                    'type': 'text',  # Specify that the input is a text type
                    'text': query,  # The actual query to be processed by the model
                }
            ],
        }
    ]

    # Print message indicating the usage of the LLM model type
    print('ModelType LLM')

    # Define additional model parameters like model name, temperature, and max tokens for LLM
    model_kwargs = {'model': 'deepseek-r1:8b', 'temperature': 1, 'max_tokens': 100}

    # Convert the input message and model kwargs into the required API parameters
    api_kwargs = custom_client.convert_inputs_to_api_kwargs(
        input=message_prompt, model_kwargs=model_kwargs, model_type=model_type
    )

    # Print the API arguments that will be passed to the call method
    print(f'api_kwargs: {api_kwargs}')

    # Call the LLM model using the prepared API arguments
    result = custom_client.call(api_kwargs, ModelType.LLM)

    # Print the result of the LLM model call (response from OpenAI)
    print(result)

    # Parse the chat completion response and output a more structured result
    response_text = custom_client.parse_chat_completion(result)

    # Print the structured response from the chat completion
    print(f'response_text: {response_text}')


if __name__ == '__main__':
    build_custom_model_client()
