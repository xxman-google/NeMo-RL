# Testing results:
# == Grading Result ==
# Question: What is the capital of France?
# Ground Truth: Paris
# Prediction: Paris is the capital of France.
# Grading response: A
# Grade: A

# install openai package if not already installed
# pip install openai
export OPENAI_API_KEY="your_openai_api_key_here"
uv run --extra mcore tests/functional/test_grader_model.py