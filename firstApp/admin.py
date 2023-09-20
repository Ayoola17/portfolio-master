from django.contrib import admin
from .models import upload
# Register your models here.

@admin.register(upload)
class RequestDemoAdmin(admin.ModelAdmin):
  list_display = [field.name for field in upload._meta.get_fields()]