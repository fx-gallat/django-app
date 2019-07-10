from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from .models import Question
from .serializers import QuestionSerializer


class BaseViewTest(APITestCase):
    client = APIClient()

    @staticmethod
    def create_question(title, description, dueDate, forceIntent , completed):
        if title != "" and description != "":
            Question.objects.create(title=title, description=description, dueDate=dueDate,
                                    forceIntent=forceIntent, completed=completed)

    def setUp(self):
        # add test data
        self.create_question("What is my name ?", "name_question", "today", False, True)
        self.create_question("What is your name ?", "name_question", "today", False, True)
        self.create_question("What is out name ?", "name_question", "today", False, True)
        self.create_question("What is their name ?", "name_question", "today", False, True)


class GetAllQuestionsTest(BaseViewTest):

    def test_get_all_questions(self):
        """
        This test ensures that all songs added in the setUp method
        exist when we make a GET request to the songs/ endpoint
        """
        # hit the API endpoint
        response = self.client.get(
            reverse("questions-all")
        )
        # fetch the data from db
        expected = Question.objects.all()
        serialized = QuestionSerializer(expected, many=True)
        self.assertEqual(response.data, serialized.data)


class PutQuestionsTest(BaseViewTest):

    def test_put_questions(self):

        # hit the API endpoint
        response = self.client.put(
            reverse("question-detail", kwargs={'id': 1}),
            data={"id": 1, "title": "what is my name", "description": "name_question",
                  "forceIntent": False, "dueDate": "today", "completed": True},
            format='json'
        )
        # fetch the data from db
        expected = Question.objects.filter(id=1)
        serialized = QuestionSerializer(expected, many=True) # return a list with one single element
        new_data ={"title": "what is my name", "description": "bot_hobby",
                   "forceIntent": False, "dueDate": "today", "completed": True},
        self.assertEqual(new_data, serialized.data[0])

    def test_put_questions_force(self):
        # hit the API endpoint
        response = self.client.put(
            reverse("question-detail", kwargs={'id': 1}),
            data={"id": 1, "title": "what is my name", "description": "name_question",
                  "forceIntent": True, "dueDate": "today", "completed": True},
            format='json'
        )
        # fetch the data from db
        expected = Question.objects.filter(id=1)
        serialized = QuestionSerializer(expected, many=True) # return a list with one single element
        self.assertEqual(response.data, serialized.data[0])
