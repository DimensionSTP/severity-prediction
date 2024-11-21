import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import json

import pandas as pd
from tqdm import tqdm

from openai import OpenAI

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="lgbm.yaml",
)
def generate_columns_mapping(
    config: DictConfig,
) -> None:
    dataset = pd.read_csv(f"{config.connected_dir}/data/{config.dataset_name}.csv")
    client = OpenAI(api_key=config.api_key)

    translated_columns = {}
    for column_name in tqdm(dataset.columns):
        translated_columns[column_name] = translate_column_name(
            column_name=column_name,
            client=client,
        )

    os.makedirs(
        f"{config.connected_dir}/metadata",
        exist_ok=True,
    )
    with open(
        f"{config.connected_dir}/{config.columns_mapping_file_path}",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            translated_columns,
            f,
            ensure_ascii=False,
            indent=4,
        )


def translate_column_name(
    column_name: str,
    client: OpenAI,
) -> str:
    prompt = f"Translate the following column name from Korean to English in one word: '{column_name}' and response only the word, without any explanation."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI chatbot, that answers questions asked by User. In particular, translate Korean well into English in the format the user wants.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=16,
            temperature=0.5,
        )
        translated_text = response.choices[0].message.content.strip()
        return translated_text
    except Exception:
        print(f"Error translating column: {column_name}.")
        return column_name


if __name__ == "__main__":
    generate_columns_mapping()
