"""
This file defines the task pipeline with structured output for sentiment extraction.
It is a series of scripts that aim to auto optimize the prompt for sentiment extraction using a small model like deepseek-r1:8bn which
proved ineffective in the first place.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union

import adalflow as adal
from adalflow.datasets.types import BaseData

TEMPLATE = r"""<START_OF_SYSTEM_MESSAGE>
    {{system_prompt}}
    {% if output_format_str is not none %}
    {{output_format_str}}
    {% endif %}
    {% if few_shot_demos is not none %}
    Here are some examples:
    {{few_shot_demos}}
    {% endif %}
    <END_OF_SYSTEM_MESSAGE>
    <START_OF_USER_MESSAGE>
    {{input_str}}
    <END_OF_USER_MESSAGE>
    """

TASK_DESC_TEMPLATE = r"""
You are an expert crypto financial analyst with deep knowledge of market dynamics and sentiment analysis.

Your task is to determine the **overall sentiment of the crypto market** from the provided news article title.
If the title shows that crypto prices are likely to rise, classify it as positive (1).

If it indicates a potential decline in prices, classify it as negative (-1). In all other cases, classify it as neutral (0).

Focus on both direct mentions and indirect implications for each asset.

If the title is irrelevant, ambiguous, or from an unknown source, output neutral (0).

{% if classes %}
{% for class in classes %}
{{loop.index-1}}. {{class.label}}, {{class.desc}}
{% endfor %}
{% endif %}

{% if few_shot_demos %}
Here are some examples:
{{few_shot_demos}}
{% endif %}

"""
#  Define classes' scores and descriptions for sentiment analysis to make the small model understand the task better
_SENTIMENT_SCORES = [1, -1, 0]
_SENTIMENT_DESCRIPTIONS = [
    'The news is likely to have a positive impact on the crypto market sentiment.',
    'The news is likely to have a negative impact on the crypto market sentiment.',
    'The news does not provide any relevant information about the crypto market sentiment.',
]
_FEW_SHOT_EXAMPLES = """
global_sentiment: -1
global_sentiment: 0
global_sentiment: 1
"""


@dataclass
class SentimentExtendedData(BaseData):
    """
    Data structure for sentiment analysis input and output.
    We will stick to the format of the raw dataset while preserving the most important fields for our task.
    Attributes:
        id (str): The unique identifier of the title.
        title (str): The title of the news article.
        global_sentiment (Literal[-1,0,1]): The global sentiment of the crypto market from the title.
    """

    id: str = field(
        default=None, metadata={'description': 'The unique identifier of the title'}
    )
    #  The title of the news article should be a feature name in the dataset
    title: str = field(
        default=None, metadata={'description': 'The title of the news article'}
    )
    global_sentiment: Literal[-1, 0, 1] = field(
        default=0,
        metadata={
            'description': 'The global sentiment of the crypto market from the title'
        },
    )
    __input_fields__ = ['title']
    __output_fields__ = ['global_sentiment']


class SentimentOutputFormat(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()
        label_desc = [
            {'label': label, 'desc': desc}
            for label, desc in zip(
                _SENTIMENT_SCORES, _SENTIMENT_DESCRIPTIONS, strict=False
            )
        ]
        task_desc_str = adal.Prompt(
            template=TASK_DESC_TEMPLATE, prompt_kwargs={'classes': label_desc}
        )()
        self.data_class = SentimentExtendedData
        self.data_class.set_task_desc(task_desc_str)
        self.parser = adal.DataClassParser(
            data_class=self.data_class, return_data_class=True, format_type='yaml'
        )
        #  Tell adalflow that it can the system prompt and the few shot demos but not the output format
        prompt_kwargs = {
            'system_prompt': adal.Parameter(
                data=self.parser.get_task_desc_str(),
                role_desc='Task description',
                requires_opt=True,
                param_type=adal.ParameterType.PROMPT,
            ),
            'output_format_str': adal.Parameter(
                data=self.parser.get_output_format_str(),
                role_desc='Output format requirements',
                requires_opt=False,
                param_type=adal.ParameterType.PROMPT,
            ),
            'few_shot_demos': adal.Parameter(
                data=None,
                requires_opt=True,  # Adalflow can add/optimize demos
                role_desc='Few shot examples for guidance',
                param_type=adal.ParameterType.DEMOS,
            ),
        }
        prompt_kwargs['few_shot_demos'].data = _FEW_SHOT_EXAMPLES
        self.llm = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=prompt_kwargs,
            template=TEMPLATE,
            output_processors=self.parser,
            use_cache=True,
        )

    def _prepare_input(self, title: str, id: Optional[str] = None) -> Dict:
        input_data = self.data_class(title=title, id=id or 'sample_0')
        input_str = self.parser.get_input_str(input_data)
        prompt_kwargs = {
            'input_str': adal.Parameter(
                data=input_str, requires_opt=False, role_desc='LLM input'
            )
        }
        return prompt_kwargs

    def call(
        self, title: str, id: Optional[str] = None
    ) -> Union[adal.GeneratorOutput, adal.Parameter]:
        prompt_kwargs = self._prepare_input(title)
        output = self.llm(prompt_kwargs=prompt_kwargs, id=id)
        return output


if __name__ == '__main__':
    # from adalflow.core.model_client import ModelClient
    from adalflow.components.model_client.ollama_client import OllamaClient

    model_client = OllamaClient()
    # model_kwargs = {
    #     "model": "deepseek-r1:8b",
    #     "stream": False,
    # }
    model_kwargs = {
        'model': 'phi3:3.8b',
        'stream': False,
    }
    sentiment_component = SentimentOutputFormat(model_client, model_kwargs)
    test_input = 'I think Bitcoin price will surge soon.'
    result = sentiment_component.call(test_input, id='btc_001')
    print('=== RESULT ===')
    print(result)
