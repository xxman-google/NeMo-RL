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

import ray
import pytest
from unittest.mock import patch
from nemo_rl.environments.math_environment import ArenaHardVerifyWorker, MathEnvConfig, MathEnvironmentMetadata


@pytest.fixture(scope="module")
def worker():
    ray.init(num_cpus=1, ignore_reinit_error=True)
    with patch('nemo_rl.environments.math_environment.GeminiGraderModel'):
        cfg: MathEnvConfig = {
            "num_workers": 1,
            "worker_type": "arena_hard",
            "grader_model_name": "gemini-2.5-pro",
        }
        # Test the underlying class directly, not the remote actor
        worker_class = ArenaHardVerifyWorker.__ray_actor_class__
        worker = worker_class(cfg)
        yield worker
    ray.shutdown()

def test_get_verdict_from_grader_response(worker: ArenaHardVerifyWorker):
    assert worker._get_verdict_from_grader_response("Blah blah [[A>B]] blah") == "A>B"
    assert worker._get_verdict_from_grader_response("Blah blah [A>B] blah") == "A>B"
    assert worker._get_verdict_from_grader_response("Blah blah [[A>>B]] blah") == "A>>B"
    assert worker._get_verdict_from_grader_response("Blah blah [[A=B]] blah") == "A=B"
    assert worker._get_verdict_from_grader_response("Blah blah [[B>A]] blah") == "B>A"
    assert worker._get_verdict_from_grader_response("Blah blah [[B>>A]] blah") == "B>>A"
    assert worker._get_verdict_from_grader_response("Blah blah [[A<B]] blah") == "A<B"
    assert worker._get_verdict_from_grader_response("Blah blah [[A<<B]] blah") == "A<<B"
    assert worker._get_verdict_from_grader_response("Blah blah [[B<A]] blah") == "B<A"
    assert worker._get_verdict_from_grader_response("Blah blah [[B<<A]] blah") == "B<<A"
    assert worker._get_verdict_from_grader_response("No verdict here") is None
    assert worker._get_verdict_from_grader_response("Multiple [[A>B]] verdicts [[B>A]]") == "B>A"
    assert worker._get_verdict_from_grader_response("[[A>B]]\n") == "A>B"


def test_get_score_from_verdict(worker: ArenaHardVerifyWorker):
    assert worker._get_score_from_verdict("A>B") == [1.0]
    assert worker._get_score_from_verdict("A>>B", weight=3) == [1.0, 1.0, 1.0]
    assert worker._get_score_from_verdict("A=B") == [0.5]
    assert worker._get_score_from_verdict("A<B") == [0.0]
    assert worker._get_score_from_verdict("A<<B", weight=3) == [0.0, 0.0, 0.0]

    assert worker._get_score_from_verdict("B>A") == [0.0]
    assert worker._get_score_from_verdict("B>>A", weight=3) == [0.0, 0.0, 0.0]
    assert worker._get_score_from_verdict("B=A") == [0.5]
    assert worker._get_score_from_verdict("B<A") == [1.0]
    assert worker._get_score_from_verdict("B<<A", weight=3) == [1.0, 1.0, 1.0]

    assert worker._get_score_from_verdict("invalid") == []


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_sample_better(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("[[A>B]]", "[[B<A]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "coding"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 1
    score, _, _ = results[0]
    # scores1 = [1 - 1.0] = [0.0], scores2 = [1.0]. avg = 0.5
    assert score == 0.5


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_baseline_better(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("[[A<B]]", "[[B>A]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "coding"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 1
    score, _, _ = results[0]
    # scores1 = [1 - 0.0] = [1.0], scores2 = [0.0]. avg = 0.5
    assert score == 0.5


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_tie(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("[[A=B]]", "[[B=A]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "coding"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 1
    score, _, _ = results[0]
    # scores1 = [1 - 0.5] = [0.5], scores2 = [0.5]. avg = 0.5
    assert score == 0.5


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_baseline_much_better(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("[[A>>B]]", "[[B>>A]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "creative_writing"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 1
    score, _, _ = results[0]
    # scores1 = [1-1, 1-1, 1-1] = [0,0,0]. scores2 = [0,0,0]. avg = 0
    assert score == 0


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_sample_much_better(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("[[A<<B]]", "[[B<<A]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "coding"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 1
    score, _, _ = results[0]
    # scores1 = [1-0, 1-0, 1-0] = [1,1,1]. scores2 = [1,1,1]. avg = 1
    assert score == 1


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_mixed_verdicts_baseline_better(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("[[B>A]]", "[[A<<B]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "coding"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 1
    score, _, _ = results[0]
    # scores1 = [1-0] = [1]. scores2 = [0,0,0]. avg = 0.25
    assert score == 0.25


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_mixed_verdicts_sample_better(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("[[A<<B]]", "[[B>A]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "coding"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 1
    score, _, _ = results[0]
    # scores1 = [1-0, 1-0, 1-0] = [1,1,1]. scores2 = [0]. avg = 0.75
    assert score == 0.75


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_invalid_verdict1(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("No verdict", "[[B>A]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "coding"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 0


@patch.object(ArenaHardVerifyWorker.__ray_actor_class__, '_grade_sample')
def test_verify_no_scores(mock_grade_sample, worker: ArenaHardVerifyWorker):
    mock_grade_sample.return_value = ("[[invalid]]", "[[invalid]]")
    pred_data = [{"prompt": "p1", "response": "r1"}]
    metadata_list: list[MathEnvironmentMetadata] = [{"baseline_model_response": "b1", "category": "coding"}]
    
    results = worker.verify(pred_data, metadata_list)
    
    assert len(results) == 0
