from django.urls import path
from ewc_app.views import QuestionListView,QuestionDetailView


urlpatterns = [
    path('api/questions/', QuestionListView.as_view(), name="questions-all"),
    path('api/question/<int:id>', QuestionDetailView.as_view(), name='question-detail'),
]
