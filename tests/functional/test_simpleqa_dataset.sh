# == Task Spec ==
# Task name: SimpleQAEval
# Prompt file: None
# System prompt file: None

# == Sample Rekeyed Data ==

# Example 1:
#   Problem: Who received the IEEE Frank Rosenblatt Award in 2010?
#   Expected Answer: Michio Sugeno
#   Subject: Science and technology
#   Category: Person

# Example 2:
#   Problem: Who was awarded the Oceanography Society's Jerlov Award in 2018?
#   Expected Answer: Annick Bricaud
#   Subject: Science and technology
#   Category: Person

# Example 3:
#   Problem: What's the name of the women's liberal arts college in Cambridge, Massachusetts?
#   Expected Answer: Radcliffe College
#   Subject: Geography
#   Category: Place

# Example 4:
#   Problem: In whose honor was the Leipzig 1877 tournament organized?
#   Expected Answer: Adolf Anderssen
#   Subject: Sports
#   Category: Person

# Example 5:
#   Problem: According to Karl Küchler, what did Empress Elizabeth of Austria's favorite sculpture depict, which was made for her villa Achilleion at Corfu?
#   Expected Answer: Poet Henrich Heine.
#   Subject: Art
#   Category: Person

python -m tests.functional.test_simpleqa_dataset