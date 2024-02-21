import os
import pandas as pd
from athina.evals import (
    DoesResponseAnswerQuery
)
from athina.loaders import RagLoader
from athina.keys import AthinaApiKey, OpenAiApiKey
OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))
 
dataset = None

def load_data():
    # Create batch dataset from list of dict objects
    raw_data = [
        {
            "query": "What is the capital of Greece?",
            "context": "Greece is often called the cradle of Western civilization.",
            "response": "Athens",
        },
        {
            "query": "What is the price of a Tesla Model 3?",
            "context": "Tesla Model 3 is a fully electric car.",
            "response": "I cannot answer this question as prices vary from country to country.",
        },
        {
            "query": "What is a shooting star?",
            "context": "Black holes are stars that have collapsed under their own gravity. They are so dense that nothing can escape their gravitational pull, not even light.",
            "response": "A shooting star is a meteor that burns up in the atmosphere.",
        }
    ]
    global dataset
    dataset = RagLoader().load_dict(raw_data)
    pd.DataFrame(dataset)

def evaluate_and_validate():
    if dataset is None:
        raise ValueError("No dataset loaded.")
    eval_model = "gpt-3.5-turbo"
    df = DoesResponseAnswerQuery(model=eval_model).run_batch(data=dataset).to_df()

    # Validation: Check if all rows in the dataframe passed the evaluation
    all_passed = df['passed'].all()
    if not all_passed:
        failed_responses = df[~df['passed']]
        print("Failed Responses:")
        print(failed_responses)
        raise ValueError("Not all responses passed the evaluation.")
    else:
        print("All responses passed the evaluation.")

if __name__ == "__main__":
    load_data()
    evaluate_and_validate()