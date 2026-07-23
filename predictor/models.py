from django.utils import timezone

from django.db import models

class LungCancerPrediction(models.Model):
    gender = models.IntegerField()
    age = models.IntegerField()
    smoking = models.IntegerField()
    yellow_fingers = models.IntegerField()
    anxiety = models.IntegerField()
    peer_pressure = models.IntegerField()
    chronic_disease = models.IntegerField()
    fatigue = models.IntegerField()
    allergy = models.IntegerField()
    wheezing = models.IntegerField()
    alcohol = models.IntegerField()
    coughing = models.IntegerField()
    short_breath = models.IntegerField()
    swallowing = models.IntegerField()
    chest_pain = models.IntegerField()
    
    result = models.CharField(max_length=100)
    

