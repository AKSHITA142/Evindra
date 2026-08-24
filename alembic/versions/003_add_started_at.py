"""Add started_at timestamp column to jobs table

Revision ID: 003_add_started_at
Revises: 002_type_strictening
Create Date: 2026-08-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '003_add_started_at'
down_revision: Union[str, None] = '002_type_strictening'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('jobs') as batch_op:
        batch_op.add_column(sa.Column('started_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('jobs') as batch_op:
        batch_op.drop_column('started_at')
