# == Eval Result ==
# Example 1
#   Ground Truth : Annick Bricaud
#   Prediction   : Dr. Annick Bricaud Honored with 2018 Jerlov Award
#   Score        : True

# Example 2
#   Ground Truth : 2023
#   Prediction   : Monash Gallery of Art Rebranded as Museum of Australian Photography in 2032
#   Score        : False


export GRADER_API_KEY="your_gemini_api_key"
uv run -v tests/functional/test_verify_worker.py