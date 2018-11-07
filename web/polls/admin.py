from django.contrib import admin
from .models import NewsInfoTable, PreprocessedNewsTable, NewsLatentSpaceTable, UserNewsTable, AuthUser
# Register your models here.

admin.site.register(NewsInfoTable)
admin.site.register(PreprocessedNewsTable)
admin.site.register(NewsLatentSpaceTable)
admin.site.register(UserNewsTable)
admin.site.register(AuthUser)
