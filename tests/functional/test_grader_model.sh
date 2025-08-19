# == Grading Result ==
# Question: What is the capital of France?
# Ground Truth: Paris
# Prediction: Paris is the capital of France.
# Grading response: A
# Grade: A

export OPENAI_API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
# pip install openai
# python -m tests.functional.test_grader_model
uv run -v tests/functional/test_grader_model.py