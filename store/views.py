from django.http.response import HttpResponse
from carts.models import Cart_Item
from carts.views import _cart_id

from django.shortcuts import get_object_or_404, render
from .models import Products
from category.models import Category
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db.models import Q

# Create your views here.


def store(request, category_slug=None):
    categories = None
    products = None

    if category_slug != None:
        categories = get_object_or_404(Category, slug=category_slug)
        products = Products.objects.filter(category=categories, is_availabel=True)
        paginator = Paginator(products, 3)
        page = request.GET.get("page")
        paged_products = paginator.get_page(page)
        products_count = products.count()
    else:
        products = (
            Products.objects.all().filter(is_availabel=True).order_by("created_date")
        )
        paginator = Paginator(products, 3)
        page = request.GET.get("page")
        paged_products = paginator.get_page(page)
        products_count = products.count()

    context = {
        "products": paged_products,
        "products_count": products_count,
    }
    return render(request, "store/store.html", context)


def product_detail(request, category_slug, product_slug):
    try:
        single_product = Products.objects.get(
            category__slug=category_slug, slug=product_slug
        )
        in_cart = Cart_Item.objects.filter(
            cart__cart_id=_cart_id(request), product=single_product
        ).exists()

    except Exception as e:
        raise e
    context = {
        "single_product": single_product,
        "in_cart": in_cart,
    }
    return render(request, "store/product_detail.html", context)


def search(request):
    if "keyword" in request.GET:
        keyword = request.GET["keyword"]
        if keyword:
            products = Products.objects.order_by("-created_date").filter(
                Q(description__icontains=keyword) | Q(product_name__icontains=keyword)
            )
            products_count = products.count()
    context = {
        "products": products,
        "products_count": products_count,
    }
    return render(request, "store/store.html", context)


from .models import MyModel
from .serializers import MyModelSerializer
from rest_framework import permissions
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import routers, serializers, viewsets

import os  # for working with files
import numpy as np  # for numerical computationss
import pandas as pd  # for working with dataframes
import torch  # Pytorch module
import matplotlib.pyplot as plt  # for plotting informations on graph and images using tensors
import torch.nn as nn  # for creating  neural networks
from torch.utils.data import DataLoader  # for dataloaders
from PIL import Image  # for checking images
import torch.nn.functional as F  # for functions for calculating loss
import torchvision.transforms as transforms  # for transforming images into tensors
from torchvision.utils import make_grid  # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from torchsummary import summary  # for getting the summary of our model
from pathlib import Path
from rest_framework.response import Response


class MyModelViewSet(viewsets.ModelViewSet):
    queryset = MyModel.objects.all()
    serializer_class = MyModelSerializer
    parser_classes = (MultiPartParser, FormParser)
    # permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def perform_create(self, serializer):
        validatedData = serializer.validated_data
        img_prediction = serializer.save()

        BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

        data_dir = BASE_DIR
        train_dir = data_dir / "train"
        valid_dir = data_dir / "valid"
        diseases = os.listdir(train_dir)
        plants = []
        NumberOfDiseases = 0
        for plant in diseases:
            if plant.split("___")[0] not in plants:
                plants.append(plant.split("___")[0])
            if plant.split("___")[1] != "healthy":
                NumberOfDiseases += 1

        # unique plants in the dataset

        # number of unique diseases
        # datasets for validation and training
        train = ImageFolder(train_dir, transform=transforms.ToTensor())
        valid = ImageFolder(valid_dir, transform=transforms.ToTensor())
        # total number of classes in train set

        # Setting the seed value
        random_seed = 7
        torch.manual_seed(random_seed)
        # setting the batch size
        batch_size = 32
        # DataLoaders for training and validation
        train_dl = DataLoader(
            train, batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)

        # for moving data into GPU (if available)
        def get_default_device():
            return torch.device("cpu")

        # for moving data to device (CPU or GPU)
        def to_device(data, device):
            """Move tensor(s) to chosen device"""
            if isinstance(data, (list, tuple)):
                return [to_device(x, device) for x in data]
            return data.to(device, non_blocking=True)

        # for loading in the device (GPU if available else CPU)
        class DeviceDataLoader:
            """Wrap a dataloader to move data to a device"""

            def __init__(self, dl, device):
                self.dl = dl
                self.device = device

            def __iter__(self):
                """Yield a batch of data after moving it to device"""
                for b in self.dl:
                    yield to_device(b, self.device)

            def __len__(self):
                """Number of batches"""
                return len(self.dl)

        device = get_default_device()

        class SimpleResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(
                    in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
                )
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(
                    in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
                )
                self.relu2 = nn.ReLU()

            def forward(self, x):
                out = self.conv1(x)
                out = self.relu1(out)
                out = self.conv2(out)
                return (
                    self.relu2(out) + x
                )  # ReLU can be applied before or after adding the input

        # for calculating the accuracy
        def accuracy(outputs, labels):
            _, preds = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(preds == labels).item() / len(preds))

        # base class for the model
        class ImageClassificationBase(nn.Module):
            def training_step(self, batch):
                images, labels = batch
                out = self(images)  # Generate predictions
                loss = F.cross_entropy(out, labels)  # Calculate loss
                return loss

            def validation_step(self, batch):
                images, labels = batch
                out = self(images)  # Generate prediction
                loss = F.cross_entropy(out, labels)  # Calculate loss
                acc = accuracy(out, labels)  # Calculate accuracy
                return {"val_loss": loss.detach(), "val_accuracy": acc}

            def validation_epoch_end(self, outputs):
                batch_losses = [x["val_loss"] for x in outputs]
                batch_accuracy = [x["val_accuracy"] for x in outputs]
                epoch_loss = torch.stack(batch_losses).mean()  # Combine loss
                epoch_accuracy = torch.stack(batch_accuracy).mean()
                return {
                    "val_loss": epoch_loss,
                    "val_accuracy": epoch_accuracy,
                }  # Combine accuracies

            def epoch_end(self, epoch, result):
                print(
                    "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                        epoch,
                        result["lrs"][-1],
                        result["train_loss"],
                        result["val_loss"],
                        result["val_accuracy"],
                    )
                )

        # Architecture for training

        # convolution block with BatchNormalization
        def ConvBlock(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(4))
            return nn.Sequential(*layers)

        # resnet architecture
        class ResNet9(ImageClassificationBase):
            def __init__(self, in_channels, num_diseases):
                super().__init__()

                self.conv1 = ConvBlock(in_channels, 64)
                self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
                self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

                self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
                self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 44
                self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

                self.classifier = nn.Sequential(
                    nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases)
                )

            def forward(self, xb):  # xb is the loaded batch
                out = self.conv1(xb)
                out = self.conv2(out)
                out = self.res1(out) + out
                out = self.conv3(out)
                out = self.conv4(out)
                out = self.res2(out) + out
                out = self.classifier(out)
                return out

        PATH = BASE_DIR / "plant-disease-model.pth"
        model = to_device(ResNet9(3, len(train.classes)), device)
        model.load_state_dict(torch.load(PATH, map_location=torch.device("cpu")))
        model.eval()

        # test_dir = "/content/drive/MyDrive/test/test"
        test_dir = BASE_DIR / "media" / "images"
        test = ImageFolder(test_dir, transform=transforms.ToTensor())
        test_images = sorted(
            os.listdir(test_dir / "leafs")
        )  # since images in test folder are in alphabetical order

        def predict_image(img, model):
            """Converts image to array and return the predicted class
            with highest probability"""
            # Convert to a batch of 1
            xb = to_device(img.unsqueeze(0), device)
            # Get predictions from model
            yb = model(xb)
            # Pick index with highest probability
            _, preds = torch.max(yb, dim=1)
            # Retrieve the class label

            return train.classes[preds[0].item()]

        # predicting first image

        for i, (img, label) in enumerate(test):
            if test_images[i] == serializer.data["image_url"].split("/")[-1]:
                formated_response = {
                    "label": test_images[i],
                    "predicted": predict_image(img, model),
                    "device_id": serializer.data["device_id"],
                }
                obj, created = MyModel.objects.update_or_create(
                    id=serializer.data["id"],
                    defaults={"prediction": formated_response["predicted"]},
                )
                return Response(formated_response)
                print(
                    "Label:", test_images[i], ", Predicted:", predict_image(img, model)
                )
        # img, label = test[0]
        # plt.imshow(img.permute(1, 2, 0))
        # print("Label:", test_images[0], ", Predicted:", predict_image(img, model))
