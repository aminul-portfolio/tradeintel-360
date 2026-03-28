from django.db import models
from django.contrib.auth.models import User


class TradingFile(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processed', 'Processed'),
        ('error', 'Error'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='trading_files',
        verbose_name='User'
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Upload Time'
    )
    file = models.FileField(
        upload_to='trading_files/',
        verbose_name='Trading File'
    )
    description = models.CharField(
        max_length=255,
        blank=True,
        help_text='Optional description of the file'
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name='Processing Status'
    )

    def __str__(self):
        return f"{self.user.username} - {self.file.name} ({self.uploaded_at.strftime('%Y-%m-%d')}) [{self.status}]"
from django.db import models

# Create your models here.
