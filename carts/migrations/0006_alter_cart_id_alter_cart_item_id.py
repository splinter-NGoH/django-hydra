# Generated by Django 4.2.2 on 2023-06-29 10:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("carts", "0005_auto_20211107_1642"),
    ]

    operations = [
        migrations.AlterField(
            model_name="cart",
            name="id",
            field=models.BigAutoField(
                auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
            ),
        ),
        migrations.AlterField(
            model_name="cart_item",
            name="id",
            field=models.BigAutoField(
                auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
            ),
        ),
    ]