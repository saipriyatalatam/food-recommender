from django.contrib import admin
from food_app.models import FoodItem, RecommendationHistory
import csv
from django.http import HttpResponse
# Register your models here.
admin.site.register(FoodItem)
admin.site.register(RecommendationHistory)


@admin.register(RecommendationHistory)
class RecommendationHistoryAdmin(admin.ModelAdmin):
    list_display = [field.name for field in RecommendationHistory._meta.fields]
    actions = ["export_as_csv"]

    def export_as_csv(self, request, queryset):
        meta = self.model._meta
        field_names = [field.name for field in meta.fields]

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename={meta}.csv'
        writer = csv.writer(response)

        writer.writerow(field_names)
        for obj in queryset:
            row = [getattr(obj, field) for field in field_names]
            writer.writerow(row)

        return response

    export_as_csv.short_description = "Export Selected as CSV"
