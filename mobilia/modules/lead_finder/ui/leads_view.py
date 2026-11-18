"""Aba Leads: tabela profissional (filtros, busca, ordenação, paginação) + painel de CRM da empresa."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHeaderView, QLineEdit,
                               QMessageBox, QSplitter, QTableView, QVBoxLayout, QWidget)

from mobilia.ui import theme

from .. import ops
from ..export import write_csv
from ..geo import UF_NAME
from ..models import PRIORITY_LABEL, STATUSES
from ..net import Cancelled
from ..site_check import check_site
from ..sources import TITLES
from ..storage import Store
from ..tasks import friendly_error
from . import icons, style
from .detail import LeadDetail
from .dialogs import ManualLeadDialog, StudioPickerDialog, open_whatsapp
from .table import COL, COLUMNS, LeadTableModel, PillDelegate, WhatsAppDelegate

PAGE_SIZES = (50, 100, 250, 500)


class LeadsView(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.offset = 0
        self.sort_key, self.sort_desc = "updated", True
        self.select_after_load: int | None = None
        self.bg_task = None
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(10)

        # filtros
        self.search = QLineEdit(); self.search.setPlaceholderText("Buscar por nome, cidade, CNPJ, site ou telefone")
        self.search.addAction(icons.icon("search", theme.MUTED), QLineEdit.LeadingPosition)
        self.search.setClearButtonEnabled(True)
        self.f_status = QComboBox(); self.f_status.addItem("Todos os status", "")
        for k, label in STATUSES:
            self.f_status.addItem(label, k)
        self.f_uf = QComboBox(); self.f_uf.addItem("Todas as UFs", "")
        self.f_source = QComboBox(); self.f_source.addItem("Todas as fontes", "")
        self.f_prio = QComboBox(); self.f_prio.addItem("Qualquer prioridade", 0)
        for k in (3, 2, 1):
            self.f_prio.addItem(f"Prioridade {PRIORITY_LABEL[k].lower()}", k)
        self.f_wa = QCheckBox("Com WhatsApp"); self.f_due = QCheckBox("Acompanhamento vencido")
        self.f_hide = QCheckBox("Ocultar “não contatar”"); self.f_hide.setChecked(True)
        lay.addLayout(style.row(self.search, self.f_status, self.f_uf, self.f_source, self.f_prio, stretch_at=0))
        self.btn_new = style.button("  Nova empresa", self._new_lead, "ghost", icons.icon("plus"))
        self.btn_export = style.button("  Exportar CSV", self._export, "outline", icons.icon("download", theme.ACCENT_HI))
        lay.addLayout(style.row(self.f_wa, self.f_due, self.f_hide, None, self.btn_new, self.btn_export))
        self._debounce = QTimer(self); self._debounce.setSingleShot(True); self._debounce.setInterval(250)
        self._debounce.timeout.connect(self.reset_and_load)
        self.search.textChanged.connect(lambda *_: self._debounce.start())
        for c in (self.f_status, self.f_uf, self.f_source, self.f_prio):
            c.currentIndexChanged.connect(lambda *_: self.reset_and_load())
        for c in (self.f_wa, self.f_due, self.f_hide):
            c.toggled.connect(lambda *_: self.reset_and_load())

        # tabela + detalhe
        self.model = LeadTableModel()
        self.table = QTableView(); self.table.setObjectName("lfTable"); self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True); self.table.setShowGrid(False); self.table.setMouseTracking(True)
        self.table.verticalHeader().hide(); self.table.verticalHeader().setDefaultSectionSize(46)
        self.table.setWordWrap(False); self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        hh = self.table.horizontalHeader()
        hh.setSectionsClickable(True); hh.setSortIndicatorShown(True); hh.setHighlightSections(False)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter); hh.setStretchLastSection(False)
        for i, (_, _, w) in enumerate(COLUMNS):
            self.table.setColumnWidth(i, w)
        hh.setSectionResizeMode(COL["name"], QHeaderView.Interactive)
        hh.setSortIndicator(-1, Qt.DescendingOrder)
        hh.sectionClicked.connect(self._sort_by)
        self.wa_delegate = WhatsAppDelegate(self.table)
        self.wa_delegate.clicked.connect(lambda lid: open_whatsapp(self.ctx, self, lid))
        self.table.setItemDelegateForColumn(COL["whatsapp"], self.wa_delegate)
        self.table.setItemDelegateForColumn(COL["status"], PillDelegate(self.table))
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)

        self.detail = LeadDetail(ctx)
        self.detail.setMinimumWidth(380)
        self.detail.whatsapp_requested.connect(lambda lid: open_whatsapp(self.ctx, self, lid))
        self.detail.site_requested.connect(lambda lid: self.analyze_sites([lid]))
        self.detail.cnpj_requested.connect(lambda lid: self.lookup_cnpj([lid]))
        self.detail.google_refresh_requested.connect(self.refresh_google)
        self.detail.studio_requested.connect(self._studio)
        split = QSplitter(Qt.Horizontal); split.setChildrenCollapsible(False)
        split.addWidget(self.table); split.addWidget(self.detail)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2); split.setSizes([900, 440])
        lay.addWidget(split, 1)

        # rodapé
        self.page_info = style.label("", "muted")
        self.page_size = QComboBox()
        for n in PAGE_SIZES:
            self.page_size.addItem(f"{n} por página", n)
        self.page_size.setCurrentIndex(max(0, PAGE_SIZES.index(ctx.settings.page_size)
                                           if ctx.settings.page_size in PAGE_SIZES else 1))
        self.page_size.currentIndexChanged.connect(lambda *_: self.reset_and_load())
        self.btn_prev = style.button("‹ Anterior", lambda: self._page(-1), "ghost")
        self.btn_next = style.button("Próxima ›", lambda: self._page(1), "ghost")
        self.sel_info = style.label("", "caption")
        self.btn_sites = style.button("  Analisar sites", lambda: self.analyze_sites(self.selected_ids()),
                                      icon=icons.icon("globe"), small=True)
        self.btn_cnpjs = style.button("  Consultar CNPJ", lambda: self.lookup_cnpj(self.selected_ids()),
                                      icon=icons.icon("building"), small=True)
        self.btn_score = style.button("  Recalcular pontuação", self._rescore_selected, icon=icons.icon("star"),
                                      small=True)
        self.btn_cancel_bg = style.button("Cancelar", self._cancel_bg, "danger", small=True)
        self.btn_cancel_bg.hide()
        lay.addLayout(style.row(self.page_info, self.page_size, self.btn_prev, self.btn_next, None, self.sel_info,
                                self.btn_sites, self.btn_cnpjs, self.btn_score, self.btn_cancel_bg))
        self._update_selection_buttons()
        ctx.leads_changed.connect(self._schedule_reload)
        self._reload_timer = QTimer(self); self._reload_timer.setSingleShot(True); self._reload_timer.setInterval(350)
        self._reload_timer.timeout.connect(self.load)

    # ---------------------------------------------------------------------------------- carregar
    def filters(self) -> dict:
        return {"text": self.search.text().strip(), "status": self.f_status.currentData(),
                "uf": self.f_uf.currentData(), "source": self.f_source.currentData(),
                "priority": self.f_prio.currentData(), "has_whatsapp": self.f_wa.isChecked(),
                "followup_due": self.f_due.isChecked(), "hide_dnc": self.f_hide.isChecked()}

    def reset_and_load(self) -> None:
        self.offset = 0
        self.load()

    def _schedule_reload(self) -> None:
        if self.isVisible():
            self._reload_timer.start()
        else:
            self._stale = True

    def load(self) -> None:
        self._stale = False
        f, limit, off = self.filters(), self.page_size.currentData(), self.offset
        key, desc = self.sort_key, self.sort_desc
        self.ctx.db.call(lambda s: (s.page(f, key, desc, off, limit), s.facets()), on_done=self._loaded,
                         on_error=lambda e: self.page_info.setText(friendly_error(e)))

    def _loaded(self, res) -> None:
        (rows, total), facets = res
        keep = set(self.selected_ids())
        current = self.detail.lead["id"] if self.detail.lead else None
        self.model.set_rows(rows, total)
        self._fill_facets(facets)
        limit = self.page_size.currentData()
        if total == 0:
            self.page_info.setText("Nenhuma empresa com esses filtros." if facets["total"] else
                                   "Nenhuma empresa ainda. Faça uma pesquisa na aba Pesquisar.")
        else:
            self.page_info.setText(f"{self.offset + 1:,}–{min(total, self.offset + limit):,} de {total:,}"
                                   .replace(",", "."))
        self.btn_prev.setEnabled(self.offset > 0)
        self.btn_next.setEnabled(self.offset + limit < total)
        target = self.select_after_load
        self.select_after_load = None
        sel = self.table.selectionModel()
        for r, row in enumerate(rows):
            if row["id"] in keep or row["id"] == target:
                sel.select(self.model.index(r, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        if target:
            self.detail.load(target)
        elif current:
            self.detail.reload()

    def _fill_facets(self, facets: dict) -> None:
        for combo, values, fmt in ((self.f_uf, facets["ufs"], lambda u: f"{u} · {UF_NAME.get(u, u)}"),
                                   (self.f_source, facets["sources"], lambda s: TITLES.get(s, s))):
            cur = combo.currentData()
            have = [combo.itemData(i) for i in range(1, combo.count())]
            if have != values:
                combo.blockSignals(True)
                while combo.count() > 1:
                    combo.removeItem(1)
                for v in values:
                    combo.addItem(fmt(v), v)
                combo.setCurrentIndex(max(0, combo.findData(cur)))
                combo.blockSignals(False)

    def _page(self, step: int) -> None:
        limit = self.page_size.currentData()
        self.offset = max(0, self.offset + step * limit)
        self.load()

    def _sort_by(self, section: int) -> None:
        key = COLUMNS[section][0]
        if key == self.sort_key:
            self.sort_desc = not self.sort_desc
        else:
            self.sort_key, self.sort_desc = key, key in ("priority", "updated")
        self.table.horizontalHeader().setSortIndicator(section, Qt.DescendingOrder if self.sort_desc
                                                       else Qt.AscendingOrder)
        self.reset_and_load()

    def show_lead(self, lead_id: int) -> None:
        """Mostra uma empresa (vindo do painel ou da pesquisa). ``0`` só recarrega a lista."""
        if lead_id:
            self.search.blockSignals(True); self.search.clear(); self.search.blockSignals(False)
            self.f_status.blockSignals(True); self.f_status.setCurrentIndex(0); self.f_status.blockSignals(False)
            self.select_after_load = lead_id
            self.detail.load(lead_id)
        self.reset_and_load()

    # ----------------------------------------------------------------------------------- seleção
    def selected_ids(self) -> list[int]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        return [self.model.lead_id(r) for r in rows if self.model.lead_id(r) is not None]

    def _selection_changed(self, *_a) -> None:
        ids = self.selected_ids()
        self._update_selection_buttons()
        if len(ids) == 1 and (not self.detail.lead or self.detail.lead["id"] != ids[0]):
            self.detail.load(ids[0])

    def _update_selection_buttons(self) -> None:
        n = len(self.selected_ids()) if self.table.selectionModel() else 0
        busy = self.bg_task is not None and self.bg_task.running()
        self.sel_info.setText(f"{n} selecionada(s)" if n else "")
        for b in (self.btn_sites, self.btn_cnpjs, self.btn_score):
            b.setEnabled(bool(n) and not busy)

    # ------------------------------------------------------------------------------ ações em lote
    def _start_bg(self, fn, label: str, on_item, on_done, on_finish=None) -> bool:
        if self.bg_task is not None and self.bg_task.running():
            self.ctx.status("Aguarde a tarefa em andamento terminar.")
            return False
        self.ctx.status(label)
        self.btn_cancel_bg.show()

        def finish(*_a):
            self.btn_cancel_bg.hide()
            self._update_selection_buttons()
            if on_finish is not None:
                on_finish()

        def done(res):
            finish(); on_done(res)

        def fail(e):
            finish(); QMessageBox.warning(self, "Lead Finder", friendly_error(e))

        def cancelled():
            finish(); self.ctx.status("Tarefa cancelada.")
        self.bg_task = self.ctx.run(fn, on_item=on_item, on_done=done, on_error=fail, on_cancel=cancelled,
                                    on_progress=lambda f, m: self.ctx.status(m) if m else None)
        self._update_selection_buttons()
        return True

    def _cancel_bg(self) -> None:
        if self.bg_task is not None:
            self.bg_task.cancel()

    def analyze_sites(self, ids: list[int]) -> None:
        if not ids:
            return
        limit = self.ctx.settings.site_check_max_per_batch
        self.ctx.db.call(ops.site_targets, ids, limit, on_done=self._run_site_checks)

    def _run_site_checks(self, targets) -> None:
        if not targets:
            self.ctx.status("Nenhuma das empresas selecionadas tem site próprio para analisar.")
            return
        http = self.ctx.http

        def work(task):
            for i, (lid, site, phones) in enumerate(targets):
                task.check()
                task.progress(i / len(targets), f"Analisando site {i + 1} de {len(targets)}…")
                task.emit((lid, check_site(http, site, phones, task)))
            return len(targets)

        def item(res):
            lid, result = res
            self.ctx.db.call(lambda s: (s.save_site_check(lid, result), ops.rescore(s, [lid])),
                             on_done=lambda _r: self.ctx.leads_changed.emit())
        self._start_bg(work, "Analisando sites (só a página principal, respeitando o robots.txt)…", item,
                       lambda n: self.ctx.status(f"{n} site(s) analisado(s)."))

    def lookup_cnpj(self, ids: list[int]) -> None:
        if ids:
            self.ctx.db.call(ops.cnpj_targets, ids, on_done=self._run_cnpj)

    def _run_cnpj(self, targets) -> None:
        if not targets:
            self.ctx.status("Nenhuma das empresas selecionadas tem CNPJ.")
            return
        src = self.ctx.sources()["brasilapi"]

        def work(task):
            errors = []
            for i, (lid, cnpj) in enumerate(targets):
                task.check()
                task.progress(i / len(targets), f"Consultando CNPJ {i + 1} de {len(targets)}…")
                try:
                    task.emit((lid, src.lookup(cnpj, task)))
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - segue para o próximo
                    errors.append(friendly_error(e))
            return errors

        def item(res):
            lid, lead = res
            self.ctx.db.call(lambda s: (s.ingest([lead]), s.add_event(lid, "cnpj_consultado", lead.situacao),
                                        ops.rescore(s, [lid])), on_done=lambda _r: self.ctx.leads_changed.emit())
        self._start_bg(work, "Consultando CNPJ na BrasilAPI…", item,
                       lambda errs: self.ctx.status("CNPJ consultado." if not errs else "; ".join(errs[:3])))

    def refresh_google(self, lead_id: int) -> None:
        """Place Details (pago). Respeita o limite mensal e registra cada requisição **antes** de enviá-la, para
        que cancelamentos e erros no meio do caminho também entrem no uso do mês."""
        def go(res):
            refs, used = res
            if not refs:
                return
            if not self.ctx.begin_paid("refresh"):
                return
            try:
                self._google_refresh(lead_id, refs, used)
            except Exception:
                self.ctx.end_paid("refresh")
                raise
        self.ctx.db.call(lambda s: (s.google_refs(lead_id), s.usage_month("google_places")[0]), on_done=go)

    def _google_refresh(self, lead_id: int, refs, used: int) -> None:
        """Roda com a reserva de operação paga já obtida; libera a reserva em qualquer saída."""
        release = lambda *_a: self.ctx.end_paid("refresh")  # noqa: E731
        src = self.ctx.sources()["google_places"]
        ok, why = src.availability()
        if not ok:
            release()
            QMessageBox.information(self, "Google Places", why)
            return
        cap = self.ctx.settings.google_monthly_request_cap
        if used + len(refs) > cap:
            release()
            QMessageBox.information(self, "Google Places",
                                    f"Limite mensal atingido ({used} de {cap} requisições). "
                                    "Ajuste o limite em Configurações.")
            return
        price = self.ctx.settings.google_price_per_1000_usd

        def work(task):
            out = []
            for ref in refs:
                task.check()
                task.emit(("usage", 1))
                lead = src.details(ref, task)
                if lead is not None:
                    out.append(lead)
            return out

        def item(it):
            if it[0] == "usage":
                self.ctx.db.call(lambda s: s.record_usage("google_places", it[1], it[1] * price / 1000))

        def done(leads):
            self.ctx.db.call(lambda s: s.ingest(leads), on_done=lambda _r: self.ctx.leads_changed.emit())
        if not self._start_bg(work, "Atualizando dados pelo Google (Place Details, consulta paga)…", item, done,
                              on_finish=release):
            release()

    def _rescore_selected(self) -> None:
        ids = self.selected_ids()
        self.ctx.db.call(ops.rescore, ids, on_done=lambda n: (self.ctx.status(f"{n} pontuação(ões) recalculada(s)."),
                                                              self.ctx.leads_changed.emit()))

    def _studio(self, lead_id: int) -> None:
        dlg = StudioPickerDialog(self.ctx, lead_id, self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        self.ctx.last_dialog = dlg
        dlg.finished.connect(lambda *_: self.detail.reload())
        dlg.open()

    def _new_lead(self) -> None:
        dlg = ManualLeadDialog(self)
        self.ctx.last_dialog = dlg

        def accepted():
            lead = dlg.lead()
            self.ctx.db.call(lambda s: (lid := s.add_manual(lead), ops.rescore(s, [lid]))[0],
                             on_done=lambda lid: (self.ctx.leads_changed.emit(), self.show_lead(lid)),
                             on_error=lambda e: QMessageBox.warning(self, "Lead Finder", friendly_error(e)))
        dlg.accepted.connect(accepted)
        dlg.open()

    def _export(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Exportar leads", str(Path.home() / "leads_7xai.csv"),
                                                  "Planilha CSV (*.csv)")
        if not path:
            return
        f = self.filters()

        def work(store: Store):
            rows, attributions, omitted = store.export_rows(f)
            write_csv(path, rows, Store.EXPORT_COLUMNS, attributions)
            return len(rows), omitted
        self.ctx.db.call(work, on_done=lambda r: self.ctx.status(
            f"{r[0]} empresa(s) exportada(s) para {Path(path).name}"
            + (f" · {r[1]} campo(s) do Google omitidos pelos termos de uso" if r[1] else "")),
            on_error=lambda e: QMessageBox.warning(self, "Exportar", friendly_error(e)))

    def on_show(self) -> None:
        if getattr(self, "_stale", True) or not self.model.rows:
            self.load()
