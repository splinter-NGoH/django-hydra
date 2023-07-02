# Generated by Django 4.2.2 on 2023-07-02 14:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("store", "0007_mymodel"),
    ]

    operations = [
        migrations.AddField(
            model_name="mymodel",
            name="prediction",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name="mymodel",
            name="device_id",
            field=models.CharField(max_length=255),
        ),
    ]
