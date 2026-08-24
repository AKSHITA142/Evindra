"""Add mission_brief column to datasets table

Revision ID: 004_add_mission_brief
Revises: 003_add_started_at
Create Date: 2026-08-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '004_add_mission_brief'
down_revision: Union[str, None] = '003_add_started_at'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('datasets') as batch_op:
        batch_op.add_column(sa.Column('mission_brief', sa.String(length=2048), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('datasets') as batch_op:
        batch_op.drop_column('mission_brief')
