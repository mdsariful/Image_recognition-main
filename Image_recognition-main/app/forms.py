
from django.forms import ModelForm
from . models import Note

class MyForm(ModelForm):
	class Meta:
		model=Note
		fields = '__all__'
		#exclude = 'firstname'
