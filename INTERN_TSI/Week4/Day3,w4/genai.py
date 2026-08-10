# ============================================================
# WEEK 4 - DAY 3
# Multi-Tenant Isolation + SQLAlchemy Registry + Sentry
# ============================================================

# ============================================================
# 1. TENANT ISOLATION — ROW LEVEL
# ============================================================

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    select
)
from sqlalchemy.orm import (
    declarative_base,
    Session
)

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, nullable=False)
    content = Column(String)


engine = create_engine("sqlite:///tenant_demo.db")

Base.metadata.create_all(engine)


# Insert data for different tenants

with Session(engine) as session:

    session.add_all([
        Document(
            tenant_id=1,
            content="Tenant 1 document"
        ),
        Document(
            tenant_id=1,
            content="Tenant 1 second document"
        ),
        Document(
            tenant_id=2,
            content="Tenant 2 document"
        )
    ])

    session.commit()


# Retrieve only the current tenant's data

current_tenant_id = 1

with Session(engine) as session:

    documents = session.scalars(
        select(Document).where(
            Document.tenant_id == current_tenant_id
        )
    ).all()

    for document in documents:
        print(document.content)


# ============================================================
# 2. TENANT ISOLATION — SCHEMA LEVEL
# ============================================================

# Schema-level isolation means each tenant has
# a separate database schema.

# Example PostgreSQL concept:

schema_tenant_1 = "tenant_1"
schema_tenant_2 = "tenant_2"

print("\nSchema-level isolation:")
print(f"{schema_tenant_1}.documents")
print(f"{schema_tenant_2}.documents")


# ============================================================
# 3. ROW LEVEL vs SCHEMA LEVEL
# ============================================================

print("""
Row-level isolation:
One table contains multiple tenants.

documents
--------------------------------
tenant_id | content
--------------------------------
1         | Tenant 1 data
2         | Tenant 2 data


Schema-level isolation:
Each tenant has a separate schema.

tenant_1.documents
tenant_2.documents
""")


# ============================================================
# 4. SQLALCHEMY DYNAMIC CLASS REGISTRY
# ============================================================

from sqlalchemy.orm import registry

mapper_registry = registry()


@mapper_registry.mapped
class User:

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String)


@mapper_registry.mapped
class Product:

    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String)


print("\nSQLAlchemy registry:")
print(mapper_registry.mappers)


# Access mapped classes dynamically

class_registry = {
    "User": User,
    "Product": Product
}

class_name = "User"

selected_class = class_registry[class_name]

print("\nSelected class:")
print(selected_class)


# ============================================================
# 5. DYNAMIC MODEL SELECTION
# ============================================================

def get_model(model_name):
    return class_registry.get(model_name)


model = get_model("Product")

print("\nDynamic model:")
print(model)


# ============================================================
# 6. SENTRY CUSTOM TAGS
# ============================================================

# Install:
# pip install sentry-sdk

import sentry_sdk

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN"
)


# Custom tags

sentry_sdk.set_tag(
    "tenant_id",
    "tenant_123"
)

sentry_sdk.set_tag(
    "environment",
    "production"
)


# ============================================================
# 7. SENTRY SPANS
# ============================================================

with sentry_sdk.start_span(
    op="database.query",
    name="Load tenant documents"
) as span:

    span.set_data(
        "tenant_id",
        "tenant_123"
    )

    # Database operation would happen here

    print("\nSentry span created")


# ============================================================
# 8. SENTRY ERROR FILTERING
# ============================================================

def before_send(event, hint):

    exception = hint.get("exc_info")

    if exception:

        error = exception[1]

        # Ignore a specific error type
        if isinstance(error, ValueError):
            return None

    return event


sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    before_send=before_send
)


print("""
Sentry error filtering:

Error
 ↓
before_send()
 ↓
Check error
 ↓
return None → discard event
return event → send to Sentry
""")


# ============================================================
# SUMMARY
# ============================================================

print("""
DAY 3 SUMMARY

1. Row-level tenant isolation
   → tenant_id filters rows.

2. Schema-level tenant isolation
   → each tenant uses a separate schema.

3. SQLAlchemy registry
   → keeps track of mapped classes and
     allows dynamic model access.

4. Sentry custom tags
   → attach searchable metadata to events.

5. Sentry spans
   → measure operations such as database queries.

6. Sentry error filtering
   → before_send() can discard unwanted errors.
""")