from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('auth/login/', views.auth_page, name='auth_page'),
    path('auth/signup/', views.signup_view, name='signup'),
    path('auth/login-submit/', views.login_view, name='login'),
    path('auth/logout/', views.logout_view, name='logout'),
]


