from django.db import models


class Question(models.Model):
    """Stores a question."""
    title = models.CharField(max_length=2000)
    description = models.CharField(max_length=5000)
    forceIntent = models.BooleanField(default=False)
    dueDate = models.CharField(max_length=2000)
    completed = models.BooleanField(default=True)

    # Meta data about the database table.
    class Meta:
        db_table = 'question'
        ordering = ['id']

    # Define what to output when the model is printed as a string.
    def __str__(self):
        return self.title

