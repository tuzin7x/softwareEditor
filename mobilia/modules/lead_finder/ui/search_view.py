"""Aba Pesquisar: onde (Brasil, região, estados, municípios), o quê (palavras-chave) e em quais fontes."""
from __future__ import annotations

from PySide6.QtCore import QStringListModel, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QCompleter, QGridLayout, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QProgressBar, QScrollArea, QStackedWidget, QToolButton, QVBoxLayout,
                               QWidget)

from mobilia.ui import theme
from mobilia.ui.design_widgets import Segmented

from .. import geo, keywords as kw
from .. import ops
from ..models import GeoScope, SearchQuery
from ..search import run_search
from ..tasks import friendly_error
from . import icons, style
from .parts import FlowLayoutWidget, KpiCard, toggle_chip

SCOPES = [("pais", "Brasil"), ("regiao", "Região"), ("uf", "Estados"), ("municipios", "Municípios")]


class SearchView(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.task = None
        self.counts = {"novos": 0, "consolidados": 0, "atualizados": 0, "ignorados_nao_contatar": 0}
        self.custom_keywords: list[str] = []
        self.kw_buttons: dict[str, QToolButton] = {}
        self.source_rows: dict[str, tuple[QCheckBox, object]] = {}
        self._pending_ingest = 0
        self._summary = None

        root = QGridLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setHorizontalSpacing(16)
        left = QWidget(); llay = QVBoxLayout(left); llay.setContentsMargins(0, 0, 6, 0); llay.setSpacing(14)
        llay.addWidget(self._where_card())
        llay.addWidget(self._keywords_card())
        llay.addStretch(1)
        scroll = QScrollArea(); scroll.setObjectName("lfScroll"); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame); scroll.setWidget(left)
        root.addWidget(scroll, 0, 0, 2, 1)
        root.addWidget(self._sources_card(), 0, 1)
        root.addWidget(self._progress_card(), 1, 1)
        root.setRowStretch(1, 1)
        root.setColumnStretch(0, 3); root.setColumnStretch(1, 2)
        ctx.settings_changed.connect(self.refresh_sources)

    # ------------------------------------------------------------------------------------- montagem
    def _where_card(self):
        f, lay = style.card("Onde pesquisar")
        self.scope = Segmented(SCOPES, "uf")
        self.scope.changed.connect(self._scope_changed)
        lay.addWidget(self.scope)
        self.scope_stack = QStackedWidget()
        # Brasil
        self.scope_stack.addWidget(style.label("Todas as 27 unidades da federação. Fontes pagas fazem uma consulta "
                                               "por estado e palavra-chave: confira a estimativa de custo.", "hint",
                                               wrap=True))
        # Região
        reg = QWidget(); rl = QVBoxLayout(reg); rl.setContentsMargins(0, 0, 0, 0)
        self.region = QComboBox()
        for k, (name, ufs) in geo.REGIONS.items():
            self.region.addItem(f"{name}  ·  {', '.join(ufs)}", k)
        self.region.setCurrentIndex(list(geo.REGIONS).index("SE"))
        self.region.currentIndexChanged.connect(lambda *_: self.update_estimate())
        rl.addWidget(self.region); rl.addStretch(1)
        self.scope_stack.addWidget(reg)
        # Estados
        ufw = QWidget(); ug = QGridLayout(ufw); ug.setContentsMargins(0, 0, 0, 0); ug.setSpacing(5)
        self.uf_buttons: dict[str, QToolButton] = {}
        for i, uf in enumerate(geo.UFS):
            b = toggle_chip(uf, uf == "SP", geo.UF_NAME[uf]); b.setFixedHeight(30)
            b.toggled.connect(lambda *_: self.update_estimate())
            self.uf_buttons[uf] = b
            ug.addWidget(b, i // 9, i % 9)
        ug.setRowStretch(3, 1)
        self.scope_stack.addWidget(ufw)
        # Municípios
        mw = QWidget(); ml = QVBoxLayout(mw); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(6)
        self.muni_uf = QComboBox()
        for uf in geo.UFS:
            self.muni_uf.addItem(f"{uf} · {geo.UF_NAME[uf]}", uf)
        self.muni_uf.setCurrentIndex(geo.UFS.index("SP"))
        self.muni_input = QLineEdit(); self.muni_input.setPlaceholderText("Digite o município e tecle Enter")
        self.muni_model = QStringListModel()
        comp = QCompleter(self.muni_model, self); comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setFilterMode(Qt.MatchContains)
        self.muni_input.setCompleter(comp)
        self.muni_input.returnPressed.connect(self.add_municipality)
        add = style.button("Adicionar", self.add_municipality, "outline")
        ml.addLayout(style.row(self.muni_uf, self.muni_input, add, stretch_at=1))
        self.muni_hint = style.label("", "caption"); ml.addWidget(self.muni_hint)
        self.muni_list = QListWidget(); self.muni_list.setObjectName("lfList"); self.muni_list.setMaximumHeight(120)
        self.muni_list.itemDoubleClicked.connect(lambda it: (self.muni_list.takeItem(self.muni_list.row(it)),
                                                             self.update_estimate()))
        ml.addWidget(self.muni_list)
        ml.addWidget(style.label("Duplo clique remove um município da lista.", "caption"))
        self.scope_stack.addWidget(mw)
        self.muni_uf.currentIndexChanged.connect(
            lambda *_: self.load_municipalities() if self.scope.value() == "municipios" else None)
        self.scope_stack.setCurrentIndex(2)
        lay.addWidget(self.scope_stack)
        return f

    def _keywords_card(self):
        f, lay = style.card("Palavras-chave")
        self.kw_flow = FlowLayoutWidget(3)
        lay.addWidget(self.kw_flow)
        self.kw_input = QLineEdit(); self.kw_input.setPlaceholderText("Nova palavra-chave (ex.: “imóveis na planta”)")
        self.kw_input.returnPressed.connect(self.save_keyword)
        lay.addLayout(style.row(self.kw_input, style.button("  Salvar", self.save_keyword, "outline",
                                                             icons.icon("plus", theme.ACCENT_HI)),
                                style.button("Todas", lambda: self._set_all_kw(True), "ghost"),
                                style.button("Nenhuma", lambda: self._set_all_kw(False), "ghost"), stretch_at=0))
        self.kw_msg = style.label("", "caption", wrap=True); lay.addWidget(self.kw_msg)
        self.build_keywords()
        return f

    def _sources_card(self):
        f, lay = style.card("Fontes autorizadas")
        self.sources_box = QVBoxLayout(); self.sources_box.setSpacing(8)
        lay.addLayout(self.sources_box)
        self.individuals = QCheckBox("CNPJ: incluir empresários individuais/MEI")
        self.individuals.setToolTip("O nome de empresários individuais costuma ser o nome de uma pessoa física (LGPD).")
        self.individuals.toggled.connect(lambda *_: self.update_estimate())
        lay.addWidget(self.individuals)
        self.estimate = style.label("", "lfWarn", wrap=True); lay.addWidget(self.estimate)
        self.btn_search = style.button("  Pesquisar", self.start, "primary", icons.icon("search", theme.ON_ACCENT))
        self.btn_cancel = style.button("Cancelar", self.cancel, "danger")
        self.btn_cancel.setEnabled(False)
        lay.addLayout(style.row(self.btn_search, self.btn_cancel, stretch_at=0))
        self.refresh_sources()
        return f

    def _progress_card(self):
        f, lay = style.card("Andamento", glow=True)
        self.step = style.label("Pronto para pesquisar.", "lfBig", wrap=True)
        lay.addWidget(self.step)
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setTextVisible(False); lay.addWidget(self.bar)
        grid = QGridLayout(); grid.setSpacing(10)
        self.k_new = KpiCard("Novas", hint="Empresas novas no CRM")
        self.k_merged = KpiCard("Consolidadas", theme.OK, "Duplicatas unidas a uma empresa já existente")
        self.k_updated = KpiCard("Atualizadas", theme.NOTICE_TEXT, "Registros já conhecidos, com dados renovados")
        self.k_dnc = KpiCard("Bloqueadas", theme.DANGER, "Ignoradas por pedido de não contato")
        for i, k in enumerate((self.k_new, self.k_merged, self.k_updated, self.k_dnc)):
            grid.addWidget(k, 0, i)
        lay.addLayout(grid)
        self.log = QListWidget(); self.log.setObjectName("lfList")
        lay.addWidget(self.log, 1)
        self.btn_results = style.button("  Ver resultados", lambda: self.ctx.open_lead.emit(0), "outline",
                                        icons.icon("funnel", theme.ACCENT_HI))
        lay.addWidget(self.btn_results)
        lay.addWidget(style.label("Os resultados de pesquisas diferentes são consolidados: a mesma empresa vinda de "
                                  "várias fontes vira um só registro, com a origem de cada dado.", "caption", wrap=True))
        return f

    def _scope_changed(self, value) -> None:
        self.scope_stack.setCurrentIndex([k for k, _ in SCOPES].index(value))
        if value == "municipios":
            self.load_municipalities()
        self.update_estimate()

    # ------------------------------------------------------------------------------- palavras-chave
    def build_keywords(self) -> None:
        checked = {k for k, b in self.kw_buttons.items() if b.isChecked()} if self.kw_buttons else {"Imobiliária"}
        self.kw_buttons = {}
        chips = []
        for k in list(kw.DEFAULT_KEYWORDS) + self.custom_keywords:
            custom = k in self.custom_keywords
            b = toggle_chip(k, k in checked, "Palavra-chave personalizada (clique direito para remover)" if custom
                            else ", ".join(kw.CNAE[c][:40] for c in (kw.rule(k) or ((),))[0]) or k)
            b.toggled.connect(lambda *_: self.update_estimate())
            if custom:
                b.setContextMenuPolicy(Qt.CustomContextMenu)
                b.customContextMenuRequested.connect(lambda *_a, word=k: self.delete_keyword(word))
            self.kw_buttons[k] = b
            chips.append(b)
        self.kw_flow.set_items(chips)
        self.update_estimate()

    def load_keywords(self) -> None:
        self.ctx.db.call(lambda s: s.custom_keywords(), on_done=self._keywords_loaded)

    def _keywords_loaded(self, words) -> None:
        self.custom_keywords = list(words)
        self.build_keywords()

    def save_keyword(self) -> None:
        text = self.kw_input.text()

        def ok(_r):
            self.kw_input.clear()
            self.kw_msg.setText(f"“{text.strip()}” salva. Nas fontes, ela busca pelo nome da empresa.")
            self.custom_keywords.append(text.strip())
            self.build_keywords()
            self.kw_buttons[text.strip()].setChecked(True)
        self.ctx.db.call(lambda s: s.add_keyword(text), on_done=ok,
                         on_error=lambda e: self.kw_msg.setText(friendly_error(e)))

    def delete_keyword(self, word: str) -> None:
        def ok(_r):
            self.custom_keywords = [k for k in self.custom_keywords if k != word]
            self.build_keywords()
            self.kw_msg.setText(f"“{word}” removida.")
        self.ctx.db.call(lambda s: s.delete_keyword(word), on_done=ok)

    def _set_all_kw(self, on: bool) -> None:
        for b in self.kw_buttons.values():
            b.blockSignals(True); b.setChecked(on); b.blockSignals(False)
        self.update_estimate()

    def selected_keywords(self) -> list[str]:
        return [k for k, b in self.kw_buttons.items() if b.isChecked()]

    # --------------------------------------------------------------------------------- municípios
    def load_municipalities(self) -> None:
        uf = self.muni_uf.currentData()
        cached = geo.cached_municipalities(self.ctx.cache_dir, uf)
        if cached:
            self.muni_model.setStringList(cached)
            self.muni_hint.setText(f"{len(cached)} municípios ({uf}) — lista oficial do IBGE")
            return
        self.muni_hint.setText("Carregando a lista oficial de municípios (IBGE)…")

        def done(names):
            if self.muni_uf.currentData() == uf:
                self.muni_model.setStringList(names)
                self.muni_hint.setText(f"{len(names)} municípios ({uf}) — lista oficial do IBGE")

        def fail(e):
            self.muni_hint.setText(f"Sem a lista do IBGE agora ({friendly_error(e)}). Digite o nome exato.")
        self.ctx.run(lambda t: geo.fetch_municipalities(self.ctx.http, uf, self.ctx.cache_dir, t),
                     name="lead-finder-ibge", on_done=done, on_error=fail)

    def add_municipality(self) -> None:
        uf = self.muni_uf.currentData()
        name = self.muni_input.text().strip()
        if not name:
            return
        known = self.muni_model.stringList()
        official = geo.match_municipality(name, known) if known else name.title()
        if known and not official:
            self.muni_hint.setText(f"“{name}” não está na lista oficial de {uf}.")
            return
        label = f"{official} - {uf}"
        if not self.muni_list.findItems(label, Qt.MatchExactly):
            it = QListWidgetItem(label); it.setData(Qt.UserRole, (official, uf))
            self.muni_list.addItem(it)
        self.muni_input.clear()
        self.update_estimate()

    # ------------------------------------------------------------------------------------- fontes
    def refresh_sources(self) -> None:
        """Monta a lista de fontes; a disponibilidade (chave, índice local de CNPJ) é verificada numa tarefa em
        segundo plano, porque envolve ler arquivos e o índice SQLite."""
        prev = {k: c.isChecked() for k, (c, _) in self.source_rows.items() if c.isEnabled()}
        known = dict(getattr(self, "_avail", {}))
        while self.sources_box.count():
            it = self.sources_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.source_rows = {}
        self._source_info: dict[str, QWidget] = {}
        srcs = {sid: src for sid, src in self.ctx.sources().items() if sid != "brasilapi"}
        for sid, src in srcs.items():
            box = QCheckBox(src.title)
            was_ok = known.get(sid, (False, ""))[0]
            box.setEnabled(was_ok)
            box.setChecked(was_ok and prev.get(sid, True))
            box.toggled.connect(lambda *_: self.update_estimate())
            w = QWidget(); wl = QVBoxLayout(w); wl.setContentsMargins(0, 0, 0, 0); wl.setSpacing(1)
            wl.addWidget(box)
            info = style.label(f"{src.description}  ·  verificando…", "caption", wrap=True)
            info.setContentsMargins(26, 0, 0, 0)
            wl.addWidget(info)
            self.sources_box.addWidget(w)
            self.source_rows[sid] = (box, src)
            self._source_info[sid] = info
        self._avail_gen = getattr(self, "_avail_gen", 0) + 1
        gen = self._avail_gen

        def done(avail: dict) -> None:
            if gen != self._avail_gen:
                return
            self._avail = avail
            for sid, (ok, why) in avail.items():
                box, src = self.source_rows[sid]
                box.blockSignals(True)
                box.setEnabled(ok)
                box.setChecked(ok and prev.get(sid, True))   # fonte paga ainda pede confirmação ao pesquisar
                box.blockSignals(False)
                self._source_info[sid].setText(f"{src.description}  ·  {why}")
            self.update_estimate()
        self.ctx.run(lambda t: {sid: src.availability() for sid, src in srcs.items()}, name="lead-finder-sources",
                     on_done=done)
        self.update_estimate()

    def build_query(self) -> SearchQuery:
        kind = self.scope.value()
        scope = GeoScope(kind=kind)
        if kind == "regiao":
            scope.region = self.region.currentData()
        elif kind == "uf":
            scope.ufs = [u for u, b in self.uf_buttons.items() if b.isChecked()]
        elif kind == "municipios":
            scope.municipalities = [tuple(self.muni_list.item(i).data(Qt.UserRole)) for i in range(self.muni_list.count())]
        sources = [sid for sid, (box, _) in self.source_rows.items() if box.isChecked() and box.isEnabled()]
        return SearchQuery(scope=scope, keywords=self.selected_keywords(), sources=sources,
                           include_individuals=self.individuals.isChecked())

    def validate(self, q: SearchQuery) -> str:
        if not q.keywords:
            return "Escolha pelo menos uma palavra-chave."
        if not geo.scope_ufs(q.scope):
            return "Escolha onde pesquisar (estado, região ou municípios)."
        if not q.sources:
            return "Escolha pelo menos uma fonte disponível."
        return ""

    def update_estimate(self) -> None:
        if not hasattr(self, "estimate"):
            return
        q = self.build_query()
        err = self.validate(q)
        if err:
            self.estimate.setText(err)
            return
        parts = []
        for sid in q.sources:
            est = self.source_rows[sid][1].estimate(q)
            if est.note:
                parts.append(f"{self.source_rows[sid][1].title}: {est.note}")
        self.estimate.setText(f"{q.scope.describe()} · {len(q.keywords)} palavra(s)-chave\n" + "\n".join(parts))

    # ------------------------------------------------------------------------------------ pesquisa
    def start(self) -> None:
        if self.task is not None and self.task.running():
            return
        q = self.build_query()
        err = self.validate(q)
        if err:
            self.estimate.setText(err)
            return
        sources = {sid: box_src[1] for sid, box_src in self.source_rows.items()}
        if "google_places" in q.sources:
            self.ctx.db.call(lambda s: s.usage_month("google_places")[0],
                             on_done=lambda used: self._confirm_and_run(q, sources, used))
        else:
            self._run(q, sources, 0)

    def _confirm_and_run(self, q, sources, used: int) -> None:
        est = sources["google_places"].estimate(q)
        cap = self.ctx.settings.google_monthly_request_cap
        if est.paid_requests:
            box = QMessageBox(QMessageBox.Question, "Consulta paga",
                              f"{est.note}.\nUso no mês: {used} de {cap} requisições.\n\nContinuar?",
                              QMessageBox.Yes | QMessageBox.No, self)
            if box.exec() != QMessageBox.Yes:
                return
        self._run(q, sources, used)

    def _run(self, q: SearchQuery, sources: dict, used: int) -> None:
        self.counts = dict.fromkeys(self.counts, 0)
        for k in (self.k_new, self.k_merged, self.k_updated, self.k_dnc):
            k.set_value(0, animate=False)
        self.log.clear()
        self.bar.setValue(0)
        self._summary = None
        self._pending_ingest = 0
        self.btn_search.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.step.setText("Pesquisando…")
        self._log(f"Pesquisa: {q.scope.describe()} · {', '.join(q.keywords[:5])}{'…' if len(q.keywords) > 5 else ''}")
        self._query = q
        self.task = self.ctx.run(lambda t: run_search(t, q, sources, used), name="lead-finder-search",
                                 on_progress=self._progress, on_item=self._item, on_done=self._done,
                                 on_error=self._error, on_cancel=self._cancelled)

    def cancel(self) -> None:
        if self.task is not None:
            self.task.cancel()
            self.step.setText("Cancelando…")
            self.btn_cancel.setEnabled(False)

    def _progress(self, frac, message) -> None:
        if frac is not None:
            self.bar.setValue(int(1000 * frac))
        if message:
            self.step.setText(message)

    def _log(self, text: str, color: str | None = None) -> None:
        it = QListWidgetItem(text)
        if color:
            it.setForeground(QColor(color))
        self.log.insertItem(0, it)

    def _item(self, item) -> None:
        kind = item[0]
        if kind == "batch":
            _, sid, leads = item
            self._pending_ingest += 1
            self._log(f"{self.source_rows[sid][1].title if sid in self.source_rows else sid}: "
                      f"{len(leads)} registro(s) recebidos")
            self.ctx.db.call(ops.ingest_and_score, leads, on_done=self._ingested, on_error=self._ingest_failed)
        elif kind == "usage":
            _, sid, units, cost = item
            self.ctx.db.call(lambda s: s.record_usage(sid, units, cost))
        elif kind == "unit_error":
            self._log(f"{item[2]}: {item[3]}", theme.DANGER)
        elif kind == "skipped":
            self._log(f"{item[1]}: {item[2]}", theme.NOTICE_TEXT)

    def _ingested(self, stats: dict) -> None:
        self._pending_ingest -= 1
        for k, v in stats.items():
            self.counts[k] = self.counts.get(k, 0) + v
        self.k_new.set_value(self.counts["novos"]); self.k_merged.set_value(self.counts["consolidados"])
        self.k_updated.set_value(self.counts["atualizados"]); self.k_dnc.set_value(self.counts["ignorados_nao_contatar"])
        self.ctx.leads_changed.emit()
        self._maybe_finish()

    def _ingest_failed(self, e) -> None:
        self._pending_ingest -= 1
        self._log(f"Erro ao gravar: {friendly_error(e)}", theme.DANGER)
        self._maybe_finish()

    def _done(self, summary: dict) -> None:
        self._summary = summary
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        if self._summary is None or self._pending_ingest > 0:
            return
        summary, self._summary = self._summary, None
        found = sum(s["found"] for s in summary.values())
        errors = sum(len(s["errors"]) for s in summary.values())
        self.step.setText(f"Concluído: {found} registro(s), {self.counts['novos']} nova(s) empresa(s)"
                          + (f", {errors} consulta(s) com erro" if errors else ""))
        self.bar.setValue(1000)
        self._reset_buttons()
        params = {"escopo": self._query.scope.describe(), "palavras": self._query.keywords,
                  "fontes": self._query.sources}
        self.ctx.db.call(lambda s: s.log_search(params, {"contagens": self.counts,
                                                         "fontes": {k: {"encontrados": v["found"],
                                                                        "erros": len(v["errors"])}
                                                                    for k, v in summary.items()}}))
        self.ctx.status(f"Lead Finder: pesquisa concluída — {self.counts['novos']} nova(s) empresa(s).")
        self.ctx.leads_changed.emit()

    def _error(self, e) -> None:
        self.step.setText(friendly_error(e))
        self._log(friendly_error(e), theme.DANGER)
        self._reset_buttons()

    def _cancelled(self) -> None:
        self.step.setText("Pesquisa cancelada. O que já chegou foi guardado.")
        self._reset_buttons()

    def _reset_buttons(self) -> None:
        self.btn_search.setEnabled(True); self.btn_cancel.setEnabled(False)

    def on_show(self) -> None:
        if not self.custom_keywords:
            self.load_keywords()
