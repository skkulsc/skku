# Generated by Django 2.1.2 on 2018-10-24 12:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0002_candidate'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='candidate',
            options={'managed': False},
        ),
    ]
