from django.urls import path
from . import views
from django.urls import path, include
from django.contrib.auth.models import User
from rest_framework import routers, serializers, viewsets

router = routers.DefaultRouter()
router.register(r"machinemodel", views.MyModelViewSet)
urlpatterns = [
    path("", include(router.urls)),
    path("products/", views.store, name="store"),
    path("category/<slug:category_slug>/", views.store, name="products_by_category"),
    path(
        "category/<slug:category_slug>/<slug:product_slug>/",
        views.product_detail,
        name="product_detail",
    ),
    path("search/", views.search, name="search"),
]
