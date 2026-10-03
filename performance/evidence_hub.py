from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

TEMPLATE_NAME = "performance/evidence_hub.html"


@login_required
@require_GET
def engineering_evidence(request: HttpRequest) -> HttpResponse:
    return render(request, TEMPLATE_NAME)
