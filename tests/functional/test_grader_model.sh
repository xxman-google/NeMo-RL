# == Grading Result ==
# Question: What is the capital of France?
# Ground Truth: Paris
# Prediction: Paris is the capital of France.
# Grading response: A
# Grade: A

export GRADER_API_KEY="your_api_key"
# pip install openai
# python -m tests.functional.test_grader_model
uv run -v tests/functional/test_grader_model.py