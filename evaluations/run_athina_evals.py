import os
import pandas as pd
from athina.evals import (
    DoesResponseAnswerQuery,
    Faithfulness
)
from athina.loaders import RagLoader
from athina.keys import AthinaApiKey, OpenAiApiKey
from src.rag_application import RagApplication

OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

dataset = None

def load_data():
    app = RagApplication(openai_api_key=os.getenv('OPENAI_API_KEY'))
    # Create batch dataset from list of dict objects
    raw_data = [
        {
            "query": "How much equity does YC take?",
            "expected_response": "Y Combinator takes a 7% equity stake in companies in return for $125,000 on a post-money SAFE, and a 1.5% equity stake in companies participating in the YC Fellowship Program in exchange for a $20,000 investment.",
        }
    ]
    
    # # or read from file
    # with open('evaluations/golden_dataset.jsonl', 'r') as file:
    #     raw_data = file.read()
    #     raw_data = raw_data.split('\n')
    #     data = []
    #     for item in raw_data:
    #         item = json.loads(item)
    #         item['context'], item['response'] = app.generate_response(item['query'])
    #         data.append(item)
    #     global dataset
    #     dataset = RagLoader().load_dict(data)
    #     pd.DataFrame(dataset)
    # for item in raw_data:
    #     item['context'], item['response'] = app.generate_response(item['query'])

    global dataset
    dataset = RagLoader().load_dict(raw_data)
    pd.DataFrame(dataset)

def evaluate_and_validate():
    if dataset is None:
        raise ValueError("No dataset loaded.")

    # Validate whether the response answers the query
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

    # Validate whether the response is faithful to the context
    df = Faithfulness(model=eval_model).run_batch(data=dataset).to_df()
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