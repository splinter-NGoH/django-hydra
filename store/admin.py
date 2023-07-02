from django.contrib import admin
from django.db import models
from .models import Products, Variation, MyModel

# Register your models here.


class ProductAdmin(admin.ModelAdmin):
    list_display = (
        "product_name",
        "price",
        "stock",
        "category",
        "created_date",
        "is_availabel",
    )
    prepopulated_fields = {"slug": ("product_name",)}


class MyModelAdmin(admin.ModelAdmin):
    list_display = ("device_id", "image_url", "prediction")


class VariationAdimn(admin.ModelAdmin):
    list_display = (
        "product",
        "variation_category",
        "variation_value",
        "created_date",
        "is_active",
    )
    list_editable = ("is_active",)
    list_filter = (
        "product",
        "variation_category",
        "variation_value",
        "created_date",
        "is_active",
    )


admin.site.register(Products, ProductAdmin)
admin.site.register(Variation, VariationAdimn)
admin.site.register(MyModel, MyModelAdmin)
