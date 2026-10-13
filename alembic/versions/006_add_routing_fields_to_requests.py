"""Add routing fields to requests table

Revision ID: 006
Revises: 005
Create Date: 2026-10-13 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("requests", sa.Column("routing_rationale", sa.Text(), nullable=True))
    op.add_column("requests", sa.Column("routing_score", sa.Float(precision=53), nullable=True))
    op.add_column("requests", sa.Column("desired_tier", sa.String(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column("requests", "desired_tier")
    op.drop_column("requests", "routing_score")
    op.drop_column("requests", "routing_rationale")
