from django.db import models
from django.db.models.base import Model
from django.db.models.deletion import CASCADE
from store.models import Products, Variation
from accounts.models import Account
# Create your models here.

class Cart (models.Model):
    cart_id = models.CharField(max_length= 250, blank=True)
    date_field = models.DateField(auto_now=False, auto_now_add=True)

    def __unicode__(self):
        return self.cart_id




class Cart_Item (models.Model):
    user = models.ForeignKey(Account, on_delete=models.CASCADE, null=True)
    product = models.ForeignKey(Products, on_delete=models.CASCADE)
    variations = models.ManyToManyField(Variation, blank=True)
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, null=True)
    quantity = models.IntegerField()
    is_active = models.BooleanField(default=True)

    def sub_total(self):
        return self.product.price * self.quantity
        
    def __unicode__(self):
        return self.product