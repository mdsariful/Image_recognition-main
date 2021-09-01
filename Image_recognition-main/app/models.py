from django.db import models

# Create your models here.
class Note(models.Model):
    
    age= models.CharField(max_length=200)
    gender = models.TextField(max_length=100)
    weight_status = models.TextField( max_length=100)
    health_condition = models.TextField( max_length=100)
    special_diet = models.TextField( max_length=100)
    
    def __str__(self):
        return '{}, {}, {}'.format(self.age, self.gender, self.weight_status,self.health_condition,self.special_diet)

class Note_ex(models.Model):
    calories= models.CharField(max_length=200)
    day = models.TextField(max_length=100)
    food_category = models.TextField( max_length=100)
    
    def __str__(self):
        return '{}, {},{}'.format(self.calories, self.day, self.food_category)
class note_image(models.Model):
    menu= models.TextField(max_length=200)
    reference_name = models.TextField(max_length=100)
    image_link = models.TextField( max_length=100)
    
    def __str__(self):
        return '{}, {},{}'.format(self.menu, self.reference_name, self.image_link)

class note_image_recognition(models.Model):
    image_link = models.TextField( max_length=200)
    
    def __str__(self):
        return '{}'.format(self.image)



