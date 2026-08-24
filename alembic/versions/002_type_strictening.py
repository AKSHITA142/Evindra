"""Type strictening for job status and json columns

Revision ID: 002_type_strictening
Revises: 001_initial_schema
Create Date: 2026-08-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from backend.models.base_model import JSONType


revision: str = '002_type_strictening'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure columns exist with proper JSON and Enum types
    with op.batch_alter_table('jobs') as batch_op:
        batch_op.alter_column('status',
               existing_type=sa.String(length=50),
               type_=sa.String(length=50),
               nullable=False)

    with op.batch_alter_table('experiments') as batch_op:
        batch_op.alter_column('pipeline',
               existing_type=sa.TEXT(),
               type_=JSONType(),
               existing_nullable=True)

    with op.batch_alter_table('knowledge_entries') as batch_op:
        batch_op.alter_column('source_experiment_ids',
               existing_type=sa.TEXT(),
               type_=JSONType(),
               existing_nullable=True)


def downgrade() -> None:
    with op.batch_alter_table('knowledge_entries') as batch_op:
        batch_op.alter_column('source_experiment_ids',
               existing_type=JSONType(),
               type_=sa.TEXT(),
               existing_nullable=True)

    with op.batch_alter_table('experiments') as batch_op:
        batch_op.alter_column('pipeline',
               existing_type=JSONType(),
               type_=sa.TEXT(),
               existing_nullable=True)

    with op.batch_alter_table('jobs') as batch_op:
        batch_op.alter_column('status',
               existing_type=sa.String(length=50),
               type_=sa.String(length=50),
               nullable=False)
