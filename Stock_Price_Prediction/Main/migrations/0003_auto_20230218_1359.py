# Generated by Django 3.2.18 on 2023-02-18 06:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Main', '0002_data'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Data',
        ),
        migrations.DeleteModel(
            name='File',
        ),
    ]
