# Generated by Django 3.2 on 2021-07-08 22:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_note_image'),
    ]

    operations = [
        migrations.RenameField(
            model_name='note_image',
            old_name='dreference_name',
            new_name='reference_name',
        ),
    ]