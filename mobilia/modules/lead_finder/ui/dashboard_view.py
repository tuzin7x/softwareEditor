"""Aba Painel: indicadores calculados sobre os dados reais do banco (funciona com zero registros)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QListWidget, QListWidgetItem, QScrollArea, QVBoxLayout, QWidget

from mobilia.ui import theme

from ..sources import TITLES
from . import style
from .detail import local_dt
from .parts import BarList, FunnelChart, KpiCard

KPI_COLORS = {"found": theme.ACCENT_HI, "qualified": theme.NOTICE_TEXT, "contacted": theme.ACCENT,
              "interested": theme.OK, "demo": "#B9A3E3", "negotiation": "#8FB3D9", "converted": "#7FD1AE"}


class DashboardView(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.data: dict | None = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setObjectName("lfScroll"); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame); outer.addWidget(scroll)
        body = QWidget(); scroll.setWidget(body)
        lay = QVBoxLayout(body); lay.setContentsMargins(0, 0, 8, 0); lay.setSpacing(14)

        self.kpis: dict[str, KpiCard] = {}
        grid = QGridLayout(); grid.setSpacing(10)
        from ..storage import DASH_STAGES
        for i, (key, title) in enumerate(DASH_STAGES):
            k = KpiCard(title, KPI_COLORS[key])
            self.kpis[key] = k
            grid.addWidget(k, 0, i)
        lay.addLayout(grid)
        self.empty = style.label("Ainda não há empresas. Use a aba Pesquisar: os indicadores se atualizam sozinhos.",
                                 "notice", wrap=True)
        lay.addWidget(self.empty)

        mid = QGridLayout(); mid.setSpacing(14)
        f, fl = style.card("Funil de vendas", glow=True)
        self.funnel = FunnelChart(); fl.addWidget(self.funnel)
        self.funnel_note = style.label("", "caption", wrap=True); fl.addWidget(self.funnel_note)
        mid.addWidget(f, 0, 0, 2, 1)
        r, rl = style.card("Próximos acompanhamentos")
        self.reminders = QListWidget(); self.reminders.setObjectName("lfList")
        self.reminders.itemDoubleClicked.connect(lambda it: self.ctx.open_lead.emit(it.data(Qt.UserRole)))
        rl.addWidget(self.reminders)
        self.overdue = style.label("", "lfWarn"); rl.addWidget(self.overdue)
        mid.addWidget(r, 0, 1)
        q, ql = style.card("Base de contatos")
        self.base_info = style.label("", "body", wrap=True); ql.addWidget(self.base_info)
        mid.addWidget(q, 1, 1)
        mid.setColumnStretch(0, 3); mid.setColumnStretch(1, 2)
        lay.addLayout(mid)

        bottom = QGridLayout(); bottom.setSpacing(14)
        u, ul = style.card("Distribuição por estado")
        self.ufs = BarList(theme.ACCENT); ul.addWidget(self.ufs)
        c, cl = style.card("Cidades com mais empresas")
        self.cities = BarList(theme.ACCENT_LO); cl.addWidget(self.cities)
        s, sl = style.card("Empresas por fonte")
        self.srcs = BarList(theme.OK); sl.addWidget(self.srcs)
        bottom.addWidget(u, 0, 0); bottom.addWidget(c, 0, 1); bottom.addWidget(s, 0, 2)
        lay.addLayout(bottom)
        lay.addStretch(1)
        ctx.leads_changed.connect(lambda: self.refresh() if self.isVisible() else None)

    def refresh(self) -> None:
        t = self.ctx.settings.qualified_threshold
        self.ctx.db.call(lambda s: s.dashboard(t), on_done=self.show_data)

    def show_data(self, d: dict) -> None:
        self.data = d
        v = d["values"]
        for key, k in self.kpis.items():
            k.set_value(v.get(key, 0))
        self.empty.setVisible(v["found"] == 0)
        self.funnel.set_data([(title, v.get(key, 0)) for key, title in d["stages"]])
        self.funnel_note.setText(
            f"Qualificadas: pontuação ≥ {self.ctx.settings.qualified_threshold}. "
            f"Sem interesse: {d['lost']} · Não contatar: {d['dnc']} · Sem pontuação: {d['unscored']}.")
        self.ufs.set_data(d["ufs"])
        self.cities.set_data(d["cities"])
        self.srcs.set_data([(TITLES.get(s, s), n) for s, n in d["sources"]])
        self.reminders.clear()
        for r in d["reminders"]:
            it = QListWidgetItem(f"{local_dt(r['due_at'])} · {r['name']}" + (f"\n{r['text']}" if r["text"] else ""))
            it.setData(Qt.UserRole, r["lead_id"])
            self.reminders.addItem(it)
        if not d["reminders"]:
            self.reminders.addItem("Nenhum acompanhamento nos próximos 7 dias.")
        self.overdue.setText(f"{d['overdue']} lembrete(s) vencido(s)." if d["overdue"] else "")
        self.base_info.setText(f"{v['found']} empresa(s) no CRM local · {d['with_whatsapp']} com WhatsApp conhecido "
                               f"(confirmado ou provável).")

    def on_show(self) -> None:
        self.refresh()
