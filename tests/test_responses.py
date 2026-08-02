import json
import unittest

from survey_llm_eval.responses import (
    openai_response_schema,
    validate_joint_content,
    validate_single_content,
)


LABELS = {"disagree": 1, "agree": 2}


class ResponseValidationTest(unittest.TestCase):
    def test_openai_schema_forbids_extra_items(self) -> None:
        schema = openai_response_schema(
            ["A1", "A2"], name="answers", minimum=1, maximum=5
        )
        body = schema["json_schema"]["schema"]
        self.assertEqual(body["required"], ["A1", "A2"])
        self.assertFalse(body["additionalProperties"])

    def test_joint_numeric_response(self) -> None:
        scores = validate_joint_content(
            json.dumps({"A1": 1, "A2": 5}),
            ["A1", "A2"],
            minimum=1,
            maximum=5,
        )
        self.assertEqual(scores, {"A1": 1, "A2": 5})

    def test_joint_response_rejects_extra_item(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected exactly"):
            validate_joint_content(
                json.dumps({"A1": 1, "A2": 2}),
                ["A1"],
                minimum=1,
                maximum=5,
            )

    def test_single_label_score_consistency(self) -> None:
        label, score = validate_single_content(
            json.dumps({"A1": {"label": "agree", "score": 2}}),
            "A1",
            minimum=1,
            maximum=2,
            label_to_score=LABELS,
        )
        self.assertEqual((label, score), ("agree", 2))


if __name__ == "__main__":
    unittest.main()
