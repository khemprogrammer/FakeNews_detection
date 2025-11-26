from django.db import models
from django.contrib.auth.models import User


class Search(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='searches')
    text = models.TextField()
    model = models.CharField(max_length=32)
    label = models.CharField(max_length=16)
    score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    @property
    def score_pct(self):
        return round(self.score * 100, 1)
