# Generated by Django 2.1.2 on 2018-11-14 17:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0012_auto_20181028_2337'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserScrapTable',
            fields=[
                ('id', models.AutoField(db_column='Id', primary_key=True, serialize=False)),
                ('directory', models.CharField(blank=True, max_length=128, null=True)),
                ('scrap_time', models.DateTimeField()),
            ],
            options={
                'db_table': 'user_scrap_table',
                'managed': False,
            },
        ),
    ]
