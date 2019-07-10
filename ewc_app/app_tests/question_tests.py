import unittest
from ewc_app.service.question_service.question_service import QuestionService


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = {"id": 1, "question": "sais tu nager ?", "intent": "", "force_intent" : "False"}
        self.qs = QuestionService(self.data)


class TestInit(TestModel):

    def test_qs_creation(self):
        self.assertIsNotNone(self.qs)

    def test_qs_process(self):
        intent = self.qs.process_question(pretrained_model=True)
        self.assertEqual(intent, 'bot_hobby')

