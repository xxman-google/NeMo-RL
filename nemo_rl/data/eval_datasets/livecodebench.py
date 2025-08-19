# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LiveCodeBench code generation dataset.

Original dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite
"""

import base64
import functools
import json
import pickle
import re
import zlib
from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

# original jsons are located in: /mnt/datasets/livecodebench/
_VERSION_TO_FILE = {
    "release_v1": "/tmp/logs/test.jsonl",
    "release_v2": "/tmp/logs/test2.jsonl",
    "release_v3": "/tmp/logs/test3.jsonl",
    "release_v4": "/tmp/logs/test4.jsonl",
    "release_v5": "/tmp/logs/test5.jsonl",
    "release_v6": "/tmp/logs/test6.jsonl",
    "release_latest": "/tmp/logs/test6.jsonl",
}
_FORMATTED_PROMPT_WITH_STARTER_CODE = """
{}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters. Also, remmeber to import any missing modules in the code.
```python
{}
```


"""
_FORMATTED_PROMPT_WITHOUT_STARTER_CODE = """
{}

You should use `sys.stdin` to get input line by line from stdin. Implement a function called `main()` which orchastrates the solution by reading inputs from stdin and writing the answer to stdout. Feel free to use additional functions as necessary. Next do NOT forget to call `main` function at the end of the program otherwise you will not be awarded any points. Please wrap the code in a markdown code block, like this:
```python
# your code here
```


"""
_BASE_IMPORTS = """
from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
# from sortedcontainers import SortedList, SortedDict, SortedSet
# import sortedcontainers
from operator import iand
import sys
"""


class LiveCodeBenchDataset:
    def __init__(
        self,
        code_exe_dir: str,
        version: Literal[
            "release_v1",
            "release_v2",
            "release_v3",
            "release_v4",
            "release_v5",
            "release_v6",
            "release_latest",
        ] = "release_latest",
        test_type: Literal["stdin", "functional"] = "stdin",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        self.code_exe_dir = code_exe_dir
        self.test_type = test_type
        ds = load_dataset(
            "json",
            data_files=_VERSION_TO_FILE[version],
            split="train",
        )
        ds = ds.filter(
            functools.partial(self._filter_based_on_test_type, testtype=test_type)
        )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="LiveCodeBench",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.code_processor

    def _filter_based_on_test_type(
        self, example: dict[str, Any], testtype: str
    ) -> bool:
        public_test_cases = json.loads(example["public_test_cases"])
        if not public_test_cases:
            try:
                private_test_cases = json.loads(example["private_test_cases"])
            except:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(
                                example["private_test_cases"].encode("utf-8")
                            )
                        )
                    )
                )
            return private_test_cases[0]["testtype"] == testtype
        return public_test_cases[0]["testtype"] == testtype

    def _extract_function_name(self, starter_code: str) -> str:
        pattern = r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        match = re.search(pattern, starter_code, re.MULTILINE)
        return match.group(1)

    def _create_functional_tests(self, data: dict[str, Any]) -> str:
        starter_code = data["starter_code"]
        public_test_cases = json.loads(data["public_test_cases"])
        try:
            private_test_cases = json.loads(data["private_test_cases"])
        except:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(data["private_test_cases"].encode("utf-8"))
                    )
                )
            )
        function_name = self._extract_function_name(starter_code)
        all_tests = []
        for test_case in public_test_cases + private_test_cases:
            output = test_case["output"]
            inputs = ",".join(test_case["input"].split("\n"))
            all_tests.append(f"assert s.{function_name}({inputs}) == {output}")
        return "\n".join(["s = Solution()"] + all_tests)

    def _create_std_tests(self, data: dict[str, Any]) -> list[dict[str, str]]:
        public_test_cases = json.loads(data["public_test_cases"])
        try:
            private_test_cases = json.loads(data["private_test_cases"])
        except:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(data["private_test_cases"].encode("utf-8"))
                    )
                )
            )
        return public_test_cases + private_test_cases

    def _rekey(self, data: dict[str, Any]):
        if self.test_type == "functional":
            return {
                "question": _FORMATTED_PROMPT_WITH_STARTER_CODE.format(
                    data["question_content"],
                    data["starter_code"],
                ),
                "tests": self._create_functional_tests(data),
                "code_exe_dir": self.code_exe_dir,
                "base_imports": _BASE_IMPORTS,
            }
        else:
            return {
                "question": _FORMATTED_PROMPT_WITHOUT_STARTER_CODE.format(
                    data["question_content"]
                ),
                "tests": self._create_std_tests(data),
                "code_exe_dir": self.code_exe_dir,
            }
