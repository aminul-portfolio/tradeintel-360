# performance/urls.py
from django.urls import path

from . import views

app_name = "performance"

urlpatterns = [
    # main pages
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload_file, name="upload_file"),
    path(
        "market-data/upload/",
        views.upload_market_data,
        name="upload_market_data",
    ),
    path("kpi/", views.kpi_report, name="kpi_report"),
    path(
        "retrieval-inspector/",
        views.retrieval_inspector,
        name="retrieval_inspector",
    ),
    path(
        "grounding-inspector/",
        views.grounding_inspector,
        name="grounding_inspector",
    ),
    path("project-plan/", views.project_one_plan, name="project_one_plan"),

    # exports
    path("export/excel/", views.export_excel, name="export_excel"),
    path("export/pdf/", views.download_pdf, name="download_pdf"),
    path("export/csv/", views.download_cleaned_csv, name="download_cleaned_csv"),
    path("export/cleaned-excel/", views.download_excel, name="download_excel"),

    # admin tools
    path("admin/files/", views.admin_all_files, name="admin_all_files"),
    path("admin/files/<int:file_id>/delete/", views.admin_delete_file, name="admin_delete_file"),

    # load a previously uploaded file into session
    path("load/<int:file_id>/", views.load_file, name="load_file"),
]
