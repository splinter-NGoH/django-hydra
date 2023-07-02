from rest_framework import serializers
from .models import MyModel


class MyModelSerializer(serializers.ModelSerializer):
    image_url = serializers.ImageField()
    created_at = serializers.ReadOnlyField()

    class Meta:
        model = MyModel
        fields = [
            "id",
            "device_id",
            "image_url",
            "prediction",
            "created_at",
        ]
