"""
Prepare the pipeline for training by setting evaluation function and loss function.
And process with prompt training for sentiment extraction.
"""

from typing import Any, Callable, Dict, Tuple

import adalflow as adal
import pandas as pd
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.generator import BackwardPassSetup
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from custom_model_client import OllamaModelClient

from news_sentiment.prompt_optimization.data_structured_output import (
    SentimentExtendedData,
    SentimentOutputFormat,
)

# class SentimentMatchAcc:
#     def compute_single_item(self, y: List[Dict], y_gt: List[Dict]) -> int:
#         """
#         Compute accuracy by exact match of the output sentiment scores.

#         Args:
#             y (List[Dict]): predicted global_sentiment score
#             y_gt (List[Dict]): ground truth global_sentiment score

#         Returns:
#             int: 1 if exact match, else 0
#         """
#         # If y or y_gt is None or string, parse or convert
#         if isinstance(y, str):
#             import yaml
#             try:
#                 y = yaml.safe_load(y).get("scores", [])
#             except Exception:
#                 y = []
#         if isinstance(y_gt, str):
#             import yaml
#             try:
#                 y_gt = yaml.safe_load(y_gt).get("scores", [])
#             except Exception:
#                 y_gt = []

#         # Default to empty list if None
#         y = y or []
#         y_gt = y_gt or []

#         # Ensure it’s list of dicts
#         if not all(isinstance(d, dict) for d in y):
#             y = []
#         if not all(isinstance(d, dict) for d in y_gt):
#             y_gt = []
#         y_sorted = sorted(y, key=lambda x: x['coin'])
#         y_gt_sorted = sorted(y_gt, key=lambda x: x['coin'])
#         return float(y_sorted == y_gt_sorted)
#     # def compute_single_item(self, y: List[Dict], y_gt: List[Dict], reason_pred=None, reason_gt=None) -> float:
#     #     """
#     #     Compute accuracy by comparing predicted vs ground-truth scores per coin.
#     #     Ignores the 'reason' column for scoring, but logs it for inspection.

#     #     Args:
#     #         y (List[Dict] or str): predicted sentiment scores
#     #         y_gt (List[Dict] or str): ground truth sentiment scores
#     #         reason_pred (str, optional): predicted reason
#     #         reason_gt (str, optional): ground truth reason

#     #     Returns:
#     #         float: fraction of correctly predicted (coin, score) pairs
#     #     """
#     #     # --- Parse predictions if they are strings ---
#     #     def parse_scores(obj):
#     #         if isinstance(obj, str):
#     #             try:
#     #                 data = yaml.safe_load(obj)
#     #                 if isinstance(data, dict) and "scores" in data:
#     #                     return data["scores"]
#     #                 elif isinstance(data, list):
#     #                     return data
#     #                 else:
#     #                     return []
#     #             except Exception:
#     #                 try:
#     #                     return ast.literal_eval(obj)
#     #                 except Exception:
#     #                     return []
#     #         elif isinstance(obj, list):
#     #             return obj
#     #         else:
#     #             return []

#     #     y_list = parse_scores(y)
#     #     y_gt_list = parse_scores(y_gt)

#     #     # --- Build coin → score maps ---
#     #     pred_map = {d["coin"]: d["score"] for d in y_list if isinstance(d, dict) and "coin" in d}
#     #     true_map = {d["coin"]: d["score"] for d in y_gt_list if isinstance(d, dict) and "coin" in d}

#     #     # --- Compute per-coin accuracy ---
#     #     total = len(true_map)
#     #     if total == 0:
#     #         return 1.0 if not pred_map else 0.0

#     #     correct = sum(1 for coin, score in true_map.items() if pred_map.get(coin) == score)
#     #     accuracy = correct / total

#     # --- Optional logging of reason for debugging ---
#     if reason_pred is not None or reason_gt is not None:
#         print("Predicted reason:", reason_pred)
#         print("Ground truth reason:", reason_gt)


#     return accuracy
def load_sentiment_dataset(file_path: str, nrows: int) -> list[SentimentExtendedData]:
    df = pd.read_csv(file_path, nrows=nrows)
    df['global_sentiment'] = df['global_sentiment'].astype(int)
    return [
        SentimentExtendedData(**{**row, 'id': str(row['id'])})
        for row in df.to_dict(orient='records')
    ]


class SentimentOutputAdal(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        teacher_model_config: Dict,
        backward_engine_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        task = SentimentOutputFormat(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type='exact_match').compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc='exact_match: 1 if str(y) == str(y_gt) else 0',
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config,
        )

    def prepare_task(self, sample: SentimentExtendedData):
        return self.task.call, {'title': sample.title, 'id': str(sample.id)}

    def prepare_eval(
        self, sample: SentimentExtendedData, y_pred: adal.GeneratorOutput
    ) -> float:
        # Extract predicted scores
        y_label = (
            y_pred.data.global_sentiment
            if y_pred and y_pred.data and y_pred.data.global_sentiment is not None
            else None
        )
        # Ground truth scores
        y_gt = sample.global_sentiment if sample.global_sentiment is not None else None
        return self.eval_fn, {'y': y_label, 'y_gt': y_gt}

    def prepare_loss(
        self, sample: SentimentExtendedData, y_pred: adal.Parameter, *args, **kwargs
    ) -> Tuple[Callable[..., Any], Dict]:
        # Extract predicted sentiment
        y_label = None

        # Prefer the parsed full_response if available
        if y_pred.full_response and y_pred.full_response.data:
            y_label = y_pred.full_response.data.global_sentiment
        elif y_pred and y_pred.data:
            y_label = getattr(y_pred.data, 'global_sentiment', None)

        # Assign for Adalflow optimization
        y_pred.eval_input = y_label

        # Ground truth parameter
        y_gt = adal.Parameter(
            name='y_gt',
            data=sample.global_sentiment,
            eval_input=sample.global_sentiment,
            requires_opt=False,
        )

        return self.loss_fn, {'kwargs': {'y': y_pred, 'y_gt': y_gt}}


def train(
    model_client: adal.ModelClient,
    model_kwargs: Dict,
    text_optimizer_model_config: Dict,
    backward_engine_model_config: Dict,
    teacher_model_config: Dict,
    train_batch_size=4,  # larger batch size is not that effective, probably because of llm's lost in the middle
    raw_shots: int = 32,
    bootstrap_shots: int = 4,
    max_steps=12,
    num_workers=4,
    strategy='constrained',
    optimization_order='sequential',
    debug=False,
    seed=None,
    tg: bool = False,
    max_proposals_per_step: int = 5,
    disable_backward=False,
    disable_backward_gradients=False,
):
    adal_component = SentimentOutputAdal(
        model_client=model_client,
        model_kwargs=model_kwargs,
        text_optimizer_model_config=text_optimizer_model_config,
        backward_engine_model_config=backward_engine_model_config,
        teacher_model_config=teacher_model_config,
    )
    backward_pass_setup = None
    if tg:
        backward_pass_setup = BackwardPassSetup(
            all_pred_at_once=False,
            compute_grad_for_errors_only=False,
        )
    print(adal_component)

    trainer = adal.Trainer(
        train_batch_size=train_batch_size,
        adaltask=adal_component,
        strategy=strategy,
        max_steps=max_steps,
        num_workers=num_workers,
        raw_shots=raw_shots,
        bootstrap_shots=bootstrap_shots,
        debug=debug,
        weighted_sampling=True,
        optimization_order=optimization_order,
        exclude_input_fields_from_bootstrap_demos=True,
        max_proposals_per_step=max_proposals_per_step,
        disable_backward=disable_backward,
        disable_backward_gradients=disable_backward_gradients,
        text_optimizers_config_kwargs={'max_past_history': 2},
    )
    trainer.set_random_seed(seed)
    print(trainer)
    # sentiment_extractor = SentimentExctractor(
    #     model="anthropic/ClaudeOpus4",
    #     base_url=None,
    #     )
    # load the dataframes from csv files defined earlier after running `load_datasets.py` script
    train_dataset = load_sentiment_dataset(
        file_path='data/training_dataset.csv', nrows=1000
    )
    val_dataset = load_sentiment_dataset(
        file_path='data/validation_dataset.csv', nrows=300
    )
    test_dataset = load_sentiment_dataset(
        file_path='data/testing_dataset.csv', nrows=300
    )
    ckpt, _ = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        debug=debug,
        backward_pass_setup=backward_pass_setup,
    )
    return ckpt


if __name__ == '__main__':
    import json
    import random

    random.seed(2025)
    import argparse

    parser = argparse.ArgumentParser()
    from ollama import Client

    parser.add_argument('--strategy', type=str, default='constrained')
    parser.add_argument('--use_tg', action='store_true')
    parser.add_argument('--max_proposals_per_step', type=int, default=5)
    parser.add_argument(
        'output_path', nargs='?', help='File path to save the checkpoint'
    )
    parser.add_argument('--disable_backward', action='store_true')
    parser.add_argument('--disable_backward_gradients', action='store_true')

    args = parser.parse_args()

    set_strategy = args.strategy
    set_output_path = args.output_path
    use_tg = args.use_tg
    max_proposals_per_step = args.max_proposals_per_step
    disable_backward = args.disable_backward
    disable_backward_gradients = args.disable_backward_gradients
    model_client = OllamaClient()
    ollama_client = Client(host='http://localhost:11435')  # local Ollama server
    ollama_model_client = OllamaModelClient(ollama_client, model_name='deepseek-r1:8b')

    # DeepSeek model configuration
    deepseek_model = {
        'model': 'deepseek-r1:8b',
        'stream': False,
        # "temperature": 1,
        # "top_p": 0.99
    }

    # Wrap in Adalflow Generator. If you have budget use gpt 4 or 3.5 turbo instead
    teacher_generator_config = {
        'model_client': model_client,
        'model_kwargs': deepseek_model,
    }
    model_kwargs = {
        'model': 'phi3:3.8b',
        'stream': False,
    }
    # WIth my RTX 4060 8GB GPU, optimization with 3.8b model can work but will be extremely slow. (Maybe it can take the whole day)
    # On the opposite, using a large model like ClaudOpus4 will be very fast, expensive but most importantly meaningless cause
    # we saw before that it handles very well edge cases with the original prompt which means it does not need auto optimization.
    # With this being said I leave the code here for further improvement or contribution.
    ckpt = train(
        model_client=model_client,
        model_kwargs=model_kwargs,
        text_optimizer_model_config=teacher_generator_config,
        backward_engine_model_config=teacher_generator_config,
        teacher_model_config=teacher_generator_config,
        debug=False,
        max_steps=12,
        strategy=set_strategy,
        optimization_order='sequential',
        seed=2025,
        tg=use_tg,
        max_proposals_per_step=max_proposals_per_step,
        disable_backward=disable_backward,
        disable_backward_gradients=disable_backward_gradients,
    )
    if set_output_path:
        with open(set_output_path, 'w') as f:
            json.dump({'ckpt': ckpt}, f)
        print(f'Checkpoint saved to {set_output_path}')
    else:
        print('No file path provided for saving the checkpoint.')
