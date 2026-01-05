"""movie URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
# 👇 1. 记得引入这个模块
from django.views.generic.base import RedirectView

urlpatterns = (
    [
        path("admin/", admin.site.urls),
        path("", include("user.urls")),
        path("api/", include("api.urls")),
        # 👇 2. 添加这一行 (这就把 /favicon.ico 指向了你刚才放文件的地方)
        path('favicon.ico', RedirectView.as_view(url='/static/images/favicon.ico.png', permanent=True)),
    ]
    + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)
    + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
)
