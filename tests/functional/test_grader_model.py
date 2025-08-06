import os
import re
from nemo_rl.evals import grader_model


def main():
    question = "What is the capital of France?"
    target = "Paris"
    predicted_answer = "Paris is the capital of France."

    grader_prompt = grader_model.QA_GRADER_TEMPLATE.format(
        question=question,
        target=target,
        predicted_answer=predicted_answer,
    )

    gpt_grader = grader_model.GptGraderModel(
        model="gpt-4o",
        api_key=os.getenv("GRADER_API_KEY"),
        system_message=grader_model.OPENAI_SYSTEM_MESSAGE_CHATGPT)
    prompt_messages = [gpt_grader._pack_message(content=grader_prompt, role="user")]
    grader_response = gpt_grader(prompt_messages)
    grading_response = grader_response.response_text
    match = re.search(r"(A|B|C)", grading_response)

    print("== Grading Result ==")
    print(f"Question: {question}")
    print(f"Ground Truth: {target}")
    print(f"Prediction: {predicted_answer}")
    print(f"Grading response: {grading_response}")
    print(f"Grade: {match.group(0) if match else 'N/A'}")


if __name__ == "__main__":
    main()