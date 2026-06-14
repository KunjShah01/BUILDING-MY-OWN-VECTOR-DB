"""create sparse_vectors and multi_vector_groups tables

Revision ID: 002
Revises: 001
Create Date: 2026-06-14

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sparse_vectors",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("collection_id", sa.String(), nullable=False, index=True),
        sa.Column("doc_id", sa.String(), nullable=False, index=True),
        sa.Column("sparse_embedding", sa.JSON(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_sv_collection", "sparse_vectors", ["collection_id"])
    op.create_index("idx_sv_doc", "sparse_vectors", ["collection_id", "doc_id"])

    op.create_table(
        "multi_vector_groups",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("group_id", sa.String(), unique=True, nullable=False, index=True),
        sa.Column("collection_id", sa.String(), nullable=False, index=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("vectors", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_mvg_collection", "multi_vector_groups", ["collection_id"])


def downgrade() -> None:
    op.drop_index("idx_mvg_collection", table_name="multi_vector_groups")
    op.drop_table("multi_vector_groups")
    op.drop_index("idx_sv_doc", table_name="sparse_vectors")
    op.drop_index("idx_sv_collection", table_name="sparse_vectors")
    op.drop_table("sparse_vectors")
