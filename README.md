# CI using Athina Evals

This repository demonstrates on how you can do a CI configuration for your RAG application using Athina Evals.

## Rag Application

A sample Rag application is provided in the `src` directory. This is a simple rag application built using llama_index. We are going add evaluations for this application using Athina Evals.

## Setting up the dependencies

Install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

additionally you need to install the `athina-evals` package by running the following command:

```bash
pip install athina-evals
```

## Running the evaluation script

Then you can run the evaluation script by running the following command:

```bash
python -m evaluations/run_athina_evals.py
```