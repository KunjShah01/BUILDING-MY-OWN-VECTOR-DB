"""create memories table

Revision ID: 001
Revises:
Create Date: 2026-06-14

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "memories",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("memory_id", sa.String(), unique=True, nullable=False, index=True),
        sa.Column("user_id", sa.String(), nullable=False, index=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("categories", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("meta_data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_memories_memory_id", "memories", ["memory_id"])
    op.create_index("idx_memories_user_id", "memories", ["user_id"])
    op.create_index(
        "idx_memories_user_categories", "memories", ["user_id", "categories"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("idx_memories_user_categories", table_name="memories")
    op.drop_index("idx_memories_user_id", table_name="memories")
    op.drop_index("idx_memories_memory_id", table_name="memories")
    op.drop_table("memories")
