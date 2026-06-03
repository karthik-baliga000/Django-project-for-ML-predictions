from django.db import models
from django.contrib.auth.models import User

class PredictionRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    leaf_length = models.FloatField()
    leaf_width = models.FloatField()
    colour_intensity = models.FloatField()
    spots_present = models.IntegerField()  # 0 or 1
    moisture_level = models.FloatField()
    texture = models.IntegerField()        # encoded
    humidity = models.FloatField()
    temperature = models.FloatField()
    soil_type = models.IntegerField()      # encoded
    predicted_label = models.CharField(max_length=100)
    best_algorithm = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.predicted_label} ({self.created_at})"