"""OpenR1 multilingual math dataset.

This dataset is created by translating the original English questions into other languages.
"""

import functools
import json
from typing import Any, Optional

from datasets import Dataset, load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

_LANG_TO_CODE = {
    'English': 'en',
    'Bengali': 'bn',
    'German': 'de',
    'Spanish': 'es',
    'Portuguese': 'pt',
    'Italian': 'it',
    'French': 'fr',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Russian': 'ru',
    'Swahili': 'sw',
    'Telugu': 'te',
    'Thai': 'th',
    'Chinese': 'zh',
}

_LANG_TO_INSTRUCTIONS = {
    "English": """Solve this math problem. Give the reasoning steps before wrapping the final answer on the last line by \\boxed{{}} tag.

{input}""",
    "Bengali": """এই গণিত সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তরটি \\boxed{{}} ট্যাগের ভিতরে শেষ লাইনে মোড়ানোর আগে যুক্তিসঙ্গত পদক্ষেপগুলি দিন।

{input}""",
    "German": """Lösen Sie diese Matheaufgabe. Geben Sie die logischen Schritte an, bevor Sie die endgültige Antwort in der letzten Zeile in einem \\boxed{{}}-Tag einschließen.

{input}""",
    "Spanish": """Resuelva este problema matemático. Dé los pasos de razonamiento antes de encerrar la respuesta final en la última línea dentro de una etiqueta \\boxed{{}}.

{input}""",
    "Portuguese": """Resolva este problema de matemática. Apresente os passos de raciocínio antes de envolver a resposta final na última linha dentro de uma tag \\boxed{{}}.

{input}""",
    "Italian": """Risolvi questo problema di matematica. Fornisci i passaggi del ragionamento prima di racchiudere la risposta finale nell'ultima riga all'interno di un tag \\boxed{{}}.

{input}""",
    "French": """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant d'encapsuler la réponse finale sur la dernière ligne à l'intérieur d'une balise \\boxed{{}}.

{input}""",
    "Japanese": """この数学の問題を解いてください。最終的な答えを最後の行で \\boxed{{}} タグで囲む前に、論理的な手順を述べてください。

{input}""",
    "Korean": """이 수학 문제를 풀어주세요. 최종 답을 마지막 줄에 \\boxed{{}} 태그 안에 넣기 전에 추론 과정을 제시하세요.

{input}""",
    "Russian": """Решите эту математическую задачу. Изложите шаги рассуждений, прежде чем заключать окончательный ответ в последней строке внутри тега \\boxed{{}}.

{input}""",
    "Swahili": """Tatua shida hii ya hesabu. Toa hatua za hoja kabla ya kufunga jibu la mwisho kwenye mstari wa mwisho ndani ya lebo ya \\boxed{{}}.

{input}""",
    "Telugu": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని చివరి పంక్తిలో \\boxed{{}} ట్యాగ్ లోపల పెట్టే ముందు తార్కిక దశలను ఇవ్వండి.

{input}""",
    "Thai": """แก้โจทย์คณิตศาสตร์นี้ ให้ขั้นตอนการให้เหตุผลก่อนห่อคำตอบสุดท้ายไว้ในบรรทัดสุดท้ายภายในแท็ก \\boxed{{}}.

{input}""",
    "Chinese": """解决这道数学题。在最后一行将最终答案放在 \\boxed{{}} 标签中之前，给出推理步骤。

{input}""",
}


def get_prompt_and_response(messages: list[dict[str, str]]) -> tuple[str, str]:
    prompts = []
    responses = []
    for msg in messages:
        if msg["role"] == "user":
            prompts.append(msg["parts"])
        elif msg["role"] == "assistant":
            responses.append(msg["parts"])
    return prompts[0], responses[0]


class OpenR1MathMultilingualDataset:
    def __init__(self, system_prompt_file: Optional[str] = None):
        rows = []
        with open('/tmp/logs/translated_math_conversations.jsonl', 'r') as fid:
            for line in fid:
                data = json.loads(line.strip())
                if data['question_type'] != 'math-word-problem':
                    continue
                row = dict()
                problem, _ = get_prompt_and_response(data['translated_conversation'])
                lang = data['language']
                row['problem'] = _LANG_TO_INSTRUCTIONS[lang].format(input=problem)
                row['expected_answer'] = data['final_answer']
                row['lang'] = _LANG_TO_CODE[lang]
                rows.append(row)
        self.rekeyed_ds = Dataset.from_list(rows)
        self.task_spec = TaskDataSpec(
            task_name="OpenR1MathMultilingual",
            prompt_file=None,
            system_prompt_file=system_prompt_file,
        )
        self.processor = functools.partial(
            processors.data_processor,
            question_key="problem",
            extra_env_info_key_maps=[
                ("expected_answer", "ground_truth"),
                ("lang", "lang"),
                ("problem", "problem"),
            ],
        )

