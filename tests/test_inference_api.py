# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

import datasets

from huggingface_hub.inference_api import InferenceApi

from .testing_utils import with_production_testing


class InferenceApiTest(unittest.TestCase):
    def read(self, filename: str) -> bytes:
        with open(filename, "rb") as f:
            bpayload = f.read()
        return bpayload

    @with_production_testing
    def test_simple_inference(self):
        api = InferenceApi("bert-base-uncased")
        inputs = "Hi, I think [MASK] is cool"
        results = api(inputs)
        self.assertIsInstance(results, list)

        result = results[0]
        self.assertIsInstance(result, dict)
        self.assertTrue("sequence" in result)
        self.assertTrue("score" in result)

    @with_production_testing
    def test_inference_with_params(self):
        api = InferenceApi("typeform/distilbert-base-uncased-mnli")
        inputs = (
            "I bought a device but it is not working and I would like to get"
            " reimbursed!"
        )
        params = {"candidate_labels": ["refund", "legal", "faq"]}
        result = api(inputs, params)
        self.assertIsInstance(result, dict)
        self.assertTrue("sequence" in result)
        self.assertTrue("scores" in result)

    @with_production_testing
    def test_inference_with_dict_inputs(self):
        api = InferenceApi("deepset/roberta-base-squad2")
        inputs = {
            "question": "What's my name?",
            "context": "My name is Clara and I live in Berkeley.",
        }
        result = api(inputs)
        self.assertIsInstance(result, dict)
        self.assertTrue("score" in result)
        self.assertTrue("answer" in result)

    @with_production_testing
    def test_inference_with_audio(self):
        api = InferenceApi("facebook/wav2vec2-base-960h")
        with self.assertWarns(FutureWarning):
            dataset = datasets.load_dataset(
                "patrickvonplaten/librispeech_asr_dummy",
                "clean",
                split="validation",
            )
        data = self.read(dataset["file"][0])
        result = api(data=data)
        self.assertIsInstance(result, dict)
        self.assertTrue("text" in result, f"We received {result} instead")

    @with_production_testing
    def test_inference_with_image(self):
        api = InferenceApi("google/vit-base-patch16-224")
        with self.assertWarns(FutureWarning):
            dataset = datasets.load_dataset(
                "Narsil/image_dummy",
                "image",
                split="test",
            )
        data = self.read(dataset["file"][0])
        result = api(data=data)
        self.assertIsInstance(result, list)
        for classification in result:
            self.assertIsInstance(classification, dict)
            self.assertTrue("score" in classification)
            self.assertTrue("label" in classification)

    @with_production_testing
    def test_inference_overriding_task(self):
        api = InferenceApi(
            "sentence-transformers/paraphrase-albert-small-v2",
            task="feature-extraction",
        )
        inputs = "This is an example again"
        result = api(inputs)
        self.assertIsInstance(result, list)

    @with_production_testing
    def test_inference_overriding_invalid_task(self):
        with self.assertRaises(
            ValueError, msg="Invalid task invalid-task. Make sure it's valid."
        ):
            InferenceApi("bert-base-uncased", task="invalid-task")

    @with_production_testing
    def test_inference_missing_input(self):
        api = InferenceApi("deepset/roberta-base-squad2")
        result = api({"question": "What's my name?"})
        self.assertIsInstance(result, dict)
        self.assertTrue("error" in result)
