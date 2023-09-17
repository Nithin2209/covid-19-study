from django.db import models

# Create your models here.
from django.db import models
import os 

# Create your models here.
class Brain(models.Model):
    image=models.ImageField(upload_to="app/static/saved")

    def Imagename(self):
        return os.path.basename(self.image.name)