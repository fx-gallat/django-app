from ewc_app.models import Question
from ewc_app.serializers import QuestionSerializer
from rest_framework.views import APIView
from rest_framework.generics import GenericAPIView
from rest_framework.mixins import ListModelMixin,CreateModelMixin
from django.http import Http404
from rest_framework.response import Response
from rest_framework import status
from ewc_app.service.question_service.question_service import QuestionService


class QuestionListView(ListModelMixin, CreateModelMixin, GenericAPIView):
    queryset = Question.objects.all()
    serializer_class = QuestionSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, *kwargs)

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)


class QuestionDetailView(APIView):
    """
    Retrieve, update or delete a question instance.
    """
    def get_object(self, id):
        try:
            return Question.objects.get(id=id)
        except Question.DoesNotExist:
            raise Http404

    def get(self, request, id, format=None):
        question = self.get_object(id)
        serializer = QuestionSerializer(question)
        return Response(serializer.data)

    def put(self, request, id, format=None):
        question = self.get_object(id)

        if not request.data["forceIntent"]:
            qs = QuestionService(request.data)
            deduced_intent = qs.process_question(pretrained_model=True)
            request.data["description"] = deduced_intent

        serializer = QuestionSerializer(question, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, id, format=None):
        question = self.get_object(id)
        question.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
