# Project Directory Structure

This document outlines the directory structure of the project.

## `diabetes_prediction`

-   `db.sqlite3`: SQLite database file.
-   `prediction/`: Django app for handling diabetes prediction logic. Contains models for storing prediction data, views for handling requests, forms for user input, and templates for rendering prediction-related pages.
    -   `views.py`: Contains the logic for handling HTTP requests related to predictions.
    -   `models.py`: Defines the database schema for prediction-related data.
    -   `urls.py`: Defines the URL patterns for the prediction app.
    -   `forms.py`: Contains forms used for user input in the prediction process.
    -   `admin.py`: Configures the Django admin interface for prediction models.
    -   `ml_models/`: Likely contains the trained machine learning models used for predictions.
    -   `migrations/`: Contains database migration files generated by Django.
    -   `static/`: Contains static files (CSS, JavaScript, images) for the prediction app.
    -   `templates/`: Contains HTML templates for the prediction app.
-   `accounts/`: Django app for managing user accounts, authentication, and authorization.
    -   `views.py`: Handles user registration, login, logout, and profile management.
    -   `models.py`: Defines the database schema for user accounts and profiles.
    -   `urls.py`: Defines URL patterns for account-related actions.
    -   `forms.py`: Contains forms for user registration, login, and profile updates.
    -   `templates/`: Contains HTML templates for account-related pages.
    -   `migrations/`: Contains database migration files.
-   `dashboard/`: Django app for displaying user-specific information or application analytics.
    -   `views.py`: Logic for fetching and displaying data on the dashboard.
    -   `urls.py`: URL patterns for dashboard pages.
    -   `models.py`: Defines any database models specific to the dashboard.
    -   `templates/`: HTML templates for the dashboard.
    -   `migrations/`: Contains database migration files.
-   `diabetes_prediction/`: Main Django project configuration directory.
    -   `settings.py`: Contains the project settings, including database configuration, installed apps, middleware, etc.
    -   `urls.py`: Root URL configuration for the project, delegating to app-specific URLs.
    -   `asgi.py`: Configuration for ASGI-compatible web servers.
    -   `wsgi.py`: Configuration for WSGI-compatible web servers.
-   `media/`: Directory for user-uploaded media files.
-   `requirements.txt`: Python package dependencies.
-   `manage.py`: Django's command-line utility for administrative tasks.

## `utils`

-   `predictor.py`: Utility script, likely for making predictions. 