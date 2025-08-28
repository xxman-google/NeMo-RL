"""OpenR1 multilingual math dataset.

This dataset is created by translating the original English questions into other languages.
"""

import functools
import io
from typing import Any, Optional

from datasets import Dataset, load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

_ALL_LANGUAGES = ["en", "bn", "de", "es", "pt", "it", "fr", "ja", "ko", "ru", "sw", "te", "th", "zh"]

_LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the answer after "Answer:".

{input}""",
    "bn": """এই গণিত সমস্যাটি সমাধান করুন। শেষ লাইনে "উত্তর:" বিন্যাসে চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিযুক্ত ধাপগুলো দিন। "উত্তর:" এর পরে উত্তর ছাড়া আর কিছু যোগ করবেন না।

{input}""",
    "de": """Löse dieses mathematische Problem. Gib die Begründungsschritte an, bevor du die endgültige Antwort auf der letzten Zeile für sich allein im Format "Antwort:" gibst. Füge nichts anderes als die Antwort nach "Antwort:" hinzu.

{input}""",
    "es": """Resuelve este problema matemático. Da los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta después de "Respuesta:".

{input}""",
    "pt": """Resolva este problema de matemática. Dê os passos de raciocínio antes de dar a resposta final na última linha, por si só, no formato de "Resposta:". Não adicione nada além da resposta depois de "Resposta:".

{input}""",
    "it": """Risolvi questo problema di matematica. Fornisci i passaggi del ragionamento prima di dare la risposta finale sull'ultima riga da sola nel formato "Risposta:". Non aggiungere nient'altro oltre alla risposta dopo "Risposta:".

{input}""",
    "fr": """Résous ce problème de maths. Donne les étapes de raisonnement avant de donner la réponse finale sur la dernière ligne toute seule au format "Réponse :". N'ajoute rien d'autre que la réponse après "Réponse :".

{input}""",
    "ja": """この数学の問題を解いてください。最終的な答えを「答え：」の形式で最後の行に単独で記載する前に、推論のステップを提示してください。「答え：」の後には答え以外は何も追加しないでください。

{input}""",
    "ko": """이 수학 문제를 풀어주세요. 마지막 줄에 "정답:" 형식으로 최종 정답만 표기하기 전에, 추론 과정을 제시해 주세요. "정답:" 뒤에는 정답 외에 아무것도 추가하지 마세요.

{input}""",
    "ru": """Решите эту математическую задачу. Представьте шаги рассуждения, прежде чем дать окончательный ответ на последней строке в формате «Ответ:». Не добавляйте ничего, кроме ответа после «Ответ:».

{input}""",
    "sw": """Tatua tatizo hili la hesabu. Toa hatua za hoja kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote isipokuwa jibu baada ya "Jibu:".

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి పంక్తిలో "జవాబు:" ఆకృతిలో అంతిమ జవాబును ఇవ్వడానికి ముందు తార్కిక దశలను ఇవ్వండి. "జవాబు:" తర్వాత జవాబు తప్ప మరేమీ జోడించవద్దు.

{input}""",
    "th": """แก้โจทย์คณิตศาสตร์นี้ แสดงขั้นตอนการให้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายเพียงบรรทัดเดียวในรูปแบบ "คำตอบ:" ห้ามเพิ่มสิ่งอื่นใดนอกจากคำตอบหลัง "คำตอบ:"

{input}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除答案之外的任何内容。

{input}""",
}


class OpenR1MathMultilingualDataset:
    def __init__(self, system_prompt_file: Optional[str] = None):
        ds = load_dataset("json", data_files="", split="train")
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="openr1_math_multilingual",
            prompt_file=None,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.math_rejection_sampling_processor

    def _get_problem(self, messages) -> str:
        problems = [msg["parts"] for msg in messages if msg["role"] == "user"]
        return problems[0]

    def _rekey(self, data: dict[str, Any]):
        lang = data["lang"]
        return {
            "problem": _LANG_TO_INSTRUCTIONS[lang].format(input=slf._get_problem(data["translated_conversation"])),
            "expected_answer": data["final_answer"],
            "lang": lang,
        }
