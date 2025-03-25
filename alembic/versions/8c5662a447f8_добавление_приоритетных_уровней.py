"""Добавление приоритетных уровней

Revision ID: 8c5662a447f8
Revises: 75acff9eaeda
Create Date: 2025-03-01 14:47:13.988140

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8c5662a447f8'
down_revision: Union[str, None] = '75acff9eaeda'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('PriorityLevelsTable',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('id_coin', sa.Integer(), nullable=False),
    sa.Column('time_frame', sa.String(), nullable=False),
    sa.Column('Level', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['id_coin'], ['CoinsTable.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('PriorityLevelsTable')
    # ### end Alembic commands ###
