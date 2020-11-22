"""
ASGI config for Web project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/asgi/
"""

import os
import sys
from pathlib import Path

from django.core.asgi import get_asgi_application

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.join(BASE_DIR, ''))

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Web.settings')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Web.settings')

application = get_asgi_application()
