# PostgreSQL Setup Instructions

## Installation

1. **Install PostgreSQL** (if not already installed):
   - Windows: Download from https://www.postgresql.org/download/windows/
   - Or use: `choco install postgresql` (if you have Chocolatey)

2. **Install Python PostgreSQL adapter**:
   ```bash
   pip install psycopg2-binary
   ```

## Database Configuration

1. **Create PostgreSQL database**:
   ```sql
   CREATE DATABASE fakenews_db;
   ```

2. **Update settings.py** (already configured):
   - Database name: `fakenews_db`
   - User: `postgres`
   - Password: `postgres` (change this in production!)
   - Host: `localhost`
   - Port: `5432`

3. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

## Fallback to SQLite

If you don't want to use PostgreSQL, you can switch back to SQLite by:
1. Commenting out the PostgreSQL configuration in `settings.py`
2. Uncommenting the SQLite configuration

## Notes

- Make sure PostgreSQL service is running before starting Django
- Change the default password in production!
- The database will be created automatically if it doesn't exist (depending on PostgreSQL permissions)

