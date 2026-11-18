from django.urls import path
from . import views

urlpatterns = [
    path("", views.halaman_utama, name="halaman_utama"),
    path("api/foto-olx/<str:id_listing>/", views.ambil_foto_olx_api, name="ambil_foto_olx_api"),
]