# Generated by Django 2.1.2 on 2018-10-24 12:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0004_auto_20181024_2119'),
    ]

    operations = [
        migrations.RenameField(
            model_name='candidate',
            old_name='intoroduction',
            new_name='introduction',
        ),
    ]
