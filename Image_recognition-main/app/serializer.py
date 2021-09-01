from rest_framework import serializers
from .models import Note, Note_ex, note_image
class NoteSerializer(serializers.ModelSerializer):
    class Meta:
        model=Note
        fields='__all__'

class NoteSerializer_image_recognition(serializers.ModelSerializer):
    class Meta:
        model=note_image
        fields='__all__'