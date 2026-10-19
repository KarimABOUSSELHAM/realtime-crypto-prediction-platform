"""
Script to load datasets in the shape of the dataclass defined in `auto_prompt_train.py` to process with prompt auto optimization.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def add_global_sentiment_and_split(
    path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset from the given path, add a global sentiment column based on positive and negative columns,
    and split the dataset into training, validation, and test sets.

    Args:
        path (str): The path of the raw dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test datasets
    """
    df = pd.read_csv(path)
    df['global_sentiment'] = np.where(
        df['positive'] > df['negative'],
        1,
        np.where(df['positive'] < df['negative'], -1, 0),
    )
    df = df[['id', 'title', 'global_sentiment']]
    output_path = os.path.join(os.path.dirname(path), 'refactored_dataset.csv')
    df.to_csv(output_path, index=False)
    # Split into train, val, test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=2025)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=2025)

    return train_df, val_df, test_df


# def build_claude_client():
#     cr = ClientRegistry()
#     cr.add_llm_client(
#         name="Claude",
#         provider="anthropic",
#         options={
#             "model": "claude-opus-4-1-20250805",
#             "temperature": 0.0,
#         },
#     )
#     cr.set_primary("Claude")
#     return cr
# def extract_sentiment(title: str, client_registry) -> SentimentScores:
#     return b.ExtractSentimentScores(
#         title,
#         {"client_registry": client_registry}
#     )
# def load_datasets_with_claude(path: str):
#     # Since the dataset has more than 200k rows, we will limit to say 100 rows in order not to exhaust the API quota
#     # and speed up the process.
#     df = pd.read_csv(path, nrows=100)
#     cr = build_claude_client()

#     processed_rows = []
#     for _, row in df.iterrows():
#         scores = []
#         reason = None
#         try:
#             result = extract_sentiment(row["title"], cr)
#             if hasattr(result, "scores"):
#                 extracted_scores = result.scores
#             elif isinstance(result, dict):
#                 extracted_scores = result.get("scores", [])
#             else:
#                 extracted_scores = []
#             if row["positive"] == 1:
#                 score = 1
#                 scores = [{"coin": s.coin, "score": score} for s in extracted_scores]
#             elif row["negative"] == 1:
#                 score = -1
#                 scores = [{"coin": s.coin, "score": score} for s in extracted_scores]
#             else:
#                 scores = []

#         except Exception as e:
#             print(f"Error extracting for row {row['id']}: {e}")
#             scores = []

#         processed_rows.append({
#             "id": row["id"],
#             "title": row["title"],
#             "reason": reason,
#             "scores": scores,
#         })

#     processed_df = pd.DataFrame(processed_rows)

#     # Split into train, val, test
#     train_df, temp_df = train_test_split(processed_df, test_size=0.3, random_state=42)
#     val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

#     return train_df, val_df, test_df


if __name__ == '__main__':
    train_df, val_df, test_df = add_global_sentiment_and_split(
        'data/cryptopanic_news.csv'
    )
    # print(train_df.head(20))
    train_df.to_csv('data/training_dataset.csv', index=False)
    val_df.to_csv('data/validation_dataset.csv', index=False)
    test_df.to_csv('data/testing_dataset.csv', index=False)
