"""Contrato do layout: lista de itens posicionados no referencial da sala.

Convenções (válidas para TODO o projeto):
- referencial da sala = Blender: Z para cima, piso em Z = 0, metros;
- todo asset é normalizado com origem no centro da base do bounding box e com a
  FRENTE voltada para -Y no espaço local (convenção da vista "Front" do Blender);
- ``yaw_deg`` gira em torno de Z (anti-horário visto de cima). Frente após o giro:
  (sin(yaw), -cos(yaw)). Para encostar na parede de normal interna n: yaw = atan2(n.x, -n.y).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field


@dataclass
class PlacedItem:
    id: str                       # id único da instância, ex.: "sofa_0"
    role: str                     # papel: sofa, armchair, coffee_table, rug, side_table, floor_lamp,
                                  # plant_large, plant_small, wall_art, tv_console, pillow, vase, books, bowl ...
    asset_id: str                 # id do catálogo: "ph:<polyhaven_id>" ou "proc:<tipo>"
    position: list[float]         # (x, y, z) da origem do asset (centro da base) na sala
    yaw_deg: float = 0.0
    scale: float = 1.0            # escala uniforme aplicada ao asset normalizado
    size: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # (largura X, profundidade Y, altura Z) final, m
    mount: str = "floor"          # floor | wall | on_top
    wall_id: int | None = None
    parent: str | None = None     # id do item de suporte (ex.: vaso sobre a mesa de centro)
    order: int = 0                # ordem na coreografia de montagem (0 = primeiro)
    params: dict = field(default_factory=dict)  # parâmetros de assets procedurais (cores, padrão...)
    # entrada na animação (PT11, ``layout/entrances.py``): drop | emerge | glide | grow | slide | unroll;
    # "" = padrão do papel. No deslize: direção do movimento no piso (unitária, referencial da sala)
    # e distância percorrida até o repouso (m)
    motion: str = ""
    motion_dir: list[float] | None = None
    motion_dist: float = 0.0

    @property
    def front(self) -> tuple[float, float]:
        y = math.radians(self.yaw_deg)
        return (math.sin(y), -math.cos(y))

    def footprint(self) -> list[list[float]]:
        """Polígono (4 cantos) da base no piso, sentido anti-horário."""
        w, d = self.size[0] / 2, self.size[1] / 2
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        pts = []
        for lx, ly in ((-w, -d), (w, -d), (w, d), (-w, d)):
            pts.append([self.position[0] + c * lx - s * ly, self.position[1] + s * lx + c * ly])
        return pts


@dataclass
class Layout:
    room_type: str
    style: str
    palette: dict
    items: list[PlacedItem]
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(s: str) -> "Layout":
        d = json.loads(s)
        d["items"] = [PlacedItem(**i) for i in d["items"]]
        return Layout(**d)


def yaw_facing_normal(nx: float, ny: float) -> float:
    """Yaw (graus) para que a frente do asset aponte na direção (nx, ny)."""
    return math.degrees(math.atan2(nx, -ny))
