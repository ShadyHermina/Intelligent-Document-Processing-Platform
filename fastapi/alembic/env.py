import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Build the connection URL from the same POSTGRES_* environment variables
# every other part of this project already uses (core/config.py,
# core/database.py, all scripts/*.py). No credentials are stored in
# alembic.ini or committed to the repo — consistent with the project's
# existing rule of never writing the DSN to disk or logs.
#
# Alembic's migration runner uses a SYNC engine (psycopg2), separate from
# the app's runtime asyncpg pool. This is deliberate and standard practice:
# migrations run once, serially, as a one-shot process — there's no benefit
# to async here, and using a sync driver keeps Alembic's own internals
# (which are not async-native) simple and unsurprising.
# ---------------------------------------------------------------------------
db_url = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
    f"{os.environ['POSTGRES_PASSWORD']}@"
    f"{os.environ['POSTGRES_HOST']}:"
    f"{os.environ['POSTGRES_PORT']}/"
    f"{os.environ['POSTGRES_DB']}"
)
config.set_main_option("sqlalchemy.url", db_url)

# add your model's MetaData object here
# for 'autogenerate' support
#
# This project has no SQLAlchemy ORM models — all queries are raw SQL via
# asyncpg (see core/database.py, every router, every agent). target_metadata
# stays None, which means `alembic revision --autogenerate` will NOT detect
# schema changes automatically. New migrations must be written by hand with
# explicit op.create_table() / op.add_column() calls — the same discipline
# the project already used in scripts/phase6_migrate.py, just now tracked
# in one place with enforced ordering and a real "current state" record.
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
