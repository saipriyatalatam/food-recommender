"""
URL configuration for food_recommender project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home, name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from food_app import views
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.home_view, name='home'),
    path('history/', views.history_view, name='history'),
    path('autocomplete/', views.autocomplete, name='autocomplete'),
    path('search/', views.search_results, name='search_results'),
    path('recommendations/<str:food_name>/', views.recommendations_view, name='recommendations'),
    path('submit_rating/', views.submit_rating, name='submit_rating'),
]