# Test results
# == Eval Result ==
# Example 1
#   Ground Truth : Annick Bricaud
#   Prediction   : Dr. Annick Bricaud Honored with 2018 Jerlov Award
#   Score        : True

# Example 2
#   Ground Truth : 2023
#   Prediction   : Monash Gallery of Art Rebranded as Museum of Australian Photography in 2032
#   Score        : False



export OPENAI_API_KEY="your_openai_api_key_here"
uv run -v --extra mcore tests/functional/test_verify_worker.py
