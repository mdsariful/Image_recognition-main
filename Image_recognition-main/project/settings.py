import os
from pathlib import Path

import django_heroku
import dj_database_url
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'h*+-7^9$6)zyfyqpgdbh_ig)_i4rzj!lgw+lma#k3k^q+22^(v'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'whitenoise.runserver_nostatic',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    # Third party apps
    'app',
]



MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'
ROOT_URLCONF = 'project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'project.wsgi.application'

#import environ

#env = environ.Env()
# reading .env file
#environ.Env.read_env()

# Raises django's ImproperlyConfigured exception if SECRET_KEY not in os.environ
#SECRET_KEY = env("SECRET_KEY", default="unsafe-secret-key")


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME':'d19or68voh0u5b',
        'USER': 'dgxnvngpdmnjnt',
        'PASSWORD':'9ef88826d0786e2af9724e36b312ab4ab410ffefbd7b0fd7582e57d54ab16d5b',
        'HOST': 'ec2-184-73-198-174.compute-1.amazonaws.com',
        'PORT': 5432,
    }
}
# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

# DATABASES = {
#      'default': {
#          'ENGINE': 'django.db.backends.postgresql',
#          'NAME': 'postgres',
#          'USER': 'postgres',
#          'PASSWORD': 'sarif2435',
#          'HOST': '',
#          'PORT': '',
#      }
#  }


# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/
#PROJECT_ROOT   =   os.path.join(os.path.abspath(__file__))
STATIC_URL = '/static/'
#STATIC_ROOT  =   os.path.join(BASE_DIR, 'staticfiles')
#STATICFILES_DIRS= (os.path.join(BASE_DIR, 'static'),)
STATIC_ROOT=os.path.join(BASE_DIR,'staticfiles')
#django_heroku.settings(locals())
