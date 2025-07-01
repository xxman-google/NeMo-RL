"""MSGM dataset."""

import functools
import io
from typing import Any, Optional

import requests
from datasets import Dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

_ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

_LANG_TO_FPATH = {
    "bn": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_bn.tsv",
    "de": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_de.tsv",
    "en": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_en.tsv",
    "es": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_es.tsv",
    "fr": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_fr.tsv",
    "ja": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_ja.tsv",
    "ru": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_ru.tsv",
    "sw": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_sw.tsv",
    "te": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_te.tsv",
    "th": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_th.tsv",
    "zh": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_zh.tsv",
}

_LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

{input}""",
    "de": """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.

{input}""",
    "es": """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".

{input}""",
    "fr": """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".

{input}""",
    "ja": """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。

{input}""",
    "ru": """Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

{input}""",
    "th": """แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:"

{input}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

{input}""",
}


class MGSMDataset:
    def __init__(self, system_prompt_file: Optional[str] = None):
        examples = []
        for lang in _ALL_LANGUAGES:
            lang_examples = self._get_lang_examples(lang)
            examples.extend(lang_examples)
        ds = Dataset.from_list(examples)
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="MGSM",
            prompt_file=None,
            system_prompt_file=system_prompt_file,
        )
        self.processor = functools.partial(
            processors.data_processor,
            question_key="problem",
            extra_env_info_key_maps=[
                ("expected_answer", "ground_truth"),
                ("lang", "lang"),
            ],
        )

    def _url_to_fileobj(self, url: str) -> Any:
        response = requests.get(url)
        response.raise_for_status()
        return io.BytesIO(response.content)

    def _get_lang_examples(self, lang: str) -> list[dict[str, str]]:
        fpath = _LANG_TO_FPATH[lang]
        examples = []
        with self._url_to_fileobj(fpath) as f:
            for raw_line in f:
                line = raw_line.decode("utf-8").strip()
                inputs, targets = line.split("\t")
                if "." in targets:
                    raise ValueError(f"targets {targets} contains a decimal point.")
                # targets = int(targets.replace(",", ""))
                examples.append({"inputs": inputs, "targets": targets, "lang": lang})
        return examples

    def _rekey(self, data: dict[str, Any]):
        lang = data["lang"]
        return {
            "problem": _LANG_TO_INSTRUCTIONS[lang].format(input=data["inputs"]),
            "expected_answer": data["targets"],
            "lang": lang,
        }
