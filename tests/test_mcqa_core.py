import unittest

from mcqa_core import make_answer, question_count, validate_predictions
from question_planner import classify
from folds import split_records


class MCQACoreTests(unittest.TestCase):
    def setUp(self):
        self.db = {
            "video_a": {"mc_question": [{"id": 0, "question": "q", "options": ["a", "b", "c"]}]}
        }

    def test_count_and_valid_submission(self):
        self.assertEqual(question_count(self.db), 1)
        prediction = {"video_a": [make_answer(self.db["video_a"]["mc_question"][0], 1)]}
        self.assertEqual(validate_predictions(prediction, self.db), [])

    def test_missing_and_mismatched_text_are_rejected(self):
        self.assertTrue(validate_predictions({"video_a": []}, self.db))
        prediction = {"video_a": [{"id": 0, "answer_id": 1, "answer": "a"}]}
        self.assertTrue(validate_predictions(prediction, self.db))

    def test_question_router_is_label_free_and_deterministic(self):
        self.assertEqual(classify({"question": "Which segment contains the action?"}).skill, "temporal")
        self.assertEqual(classify({"question": "How many cups are on the table?"}).skill, "counting")
        self.assertEqual(classify({"question": "What sound is heard?"}).skill, "audio")

    def test_training_folds_are_video_disjoint_and_deterministic(self):
        db = {f"r{i}": {"metadata": {"video_id": f"video_{i}"}, "mc_question": []} for i in range(20)}
        train, dev = split_records(db, dev_fraction=0.25)
        self.assertEqual(set(train) & set(dev), set())
        train2, dev2 = split_records(db, dev_fraction=0.25)
        self.assertEqual(train, train2)
        self.assertEqual(dev, dev2)

    def test_unexpected_and_duplicate_questions_are_rejected(self):
        prediction = {
            "video_a": [
                {"id": 0, "answer_id": 0, "answer": "a"},
                {"id": 0, "answer_id": 0, "answer": "a"},
            ]
        }
        errors = validate_predictions(prediction, self.db)
        self.assertTrue(any("duplicate" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
