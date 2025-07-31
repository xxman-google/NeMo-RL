"""MBPP dataset and its variants."""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class MBPPDataset:
    def __init__(
        self,
        code_exe_dir: str,
        variant: Literal["full", "sanitized"] = "full",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        self.code_exe_dir = code_exe_dir
        filename = "mbpp.jsonl" if variant == "full" else "sanitized-mbpp.json"
        file_path = f"https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/mbpp/{filename}"
        ds = load_dataset("json", data_files=file_path, split="train")
        self._template = "{question} Your code should pass these tests:\n{unit_tests}\n"
        if variant == "sanitized":
            self.rekeyed_ds = ds.map(
                self._rekey_sanitized, remove_columns=ds.column_names
            )
        else:
            self.rekeyed_ds = ds.map(self._rekey_full, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name=f"MBPP_{variant}",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.code_processor

    def _rekey_full(self, data: dict[str, Any]):
        tests = data["challenge_test_list"] + data["test_list"]
        tests = "\n".join(tests)
        question = self._template.format(question=data["text"], unit_tests=tests)
        if data["test_setup_code"]:
            tests = "\n".join([data["test_setup_code"][0], tests])
        return {
            "question": question,
            "tests": tests,
            "code_exe_dir": self.code_exe_dir,
        }

    def _rekey_sanitized(self, data: dict[str, Any]):
        tests = data["test_list"]
        tests = "\n".join(tests)
        question = self._template.format(
            question=data["prompt"], unit_tests="\n".join(tests)
        )
        if data["test_imports"]:
            tests = "\n".join([data["test_imports"][0], tests])
        return {
            "question": question,
            "tests": tests,
            "code_exe_dir": self.code_exe_dir,
        }
