# Generated by Django 2.1.2 on 2018-10-28 13:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0009_auto_20181026_1614'),
    ]

    operations = [
        migrations.CreateModel(
            name='PollsCandidate',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=10)),
                ('introduction', models.TextField()),
                ('area', models.CharField(max_length=15)),
                ('party_number', models.IntegerField()),
            ],
            options={
                'db_table': 'polls_candidate',
                'managed': False,
            },
        ),
        migrations.DeleteModel(
            name='Candidate',
        ),
        migrations.AlterModelOptions(
            name='usernewstable',
            options={'managed': False},
        ),
    ]
