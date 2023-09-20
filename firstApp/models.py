from django.db import models

# Create your models here.
class upload(models.Model):
    title=models.CharField(max_length=50)
    upload=models.FileField(upload_to="media")