# Generated by Django 4.2.2 on 2023-06-29 10:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("store", "0005_products_variationfield"),
    ]

    operations = [
        migrations.AlterField(
            model_name="products",
            name="id",
            field=models.BigAutoField(
                auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
            ),
        ),
        migrations.AlterField(
            model_name="variation",
            name="id",
            field=models.BigAutoField(
                auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
            ),
        ),
    ]
