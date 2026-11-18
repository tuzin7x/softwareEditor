"""Aba Configurações: identidade, fontes e chaves, limites de custo, importação da base de CNPJ, IA e LGPD."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QLineEdit,
                               QMessageBox, QProgressBar, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from mobilia.ui import theme

from ..config import mask
from ..geo import UF_NAME
from ..sources.cnpj_open import INDEX_NAME, OFFICIAL_PAGE, import_open_data, index_info
from ..tasks import friendly_error
from . import icons, style


def _spin(lo: int, hi: int, value: int) -> QSpinBox:
    s = QSpinBox(); s.setRange(lo, hi); s.setValue(value); s.setButtonSymbols(QSpinBox.NoButtons)
    return s


class SettingsView(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.import_task = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setObjectName("lfScroll"); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame); outer.addWidget(scroll)
        body = QWidget(); scroll.setWidget(body)
        grid = QGridLayout(body); grid.setContentsMargins(0, 0, 8, 0); grid.setSpacing(14)
        s = ctx.settings

        # identidade
        f, lay = style.card("Quem envia as mensagens")
        form = QFormLayout(); form.setSpacing(8)
        self.sender = QLineEdit(s.sender_name); self.sender.setPlaceholderText("Seu nome (opcional)")
        self.company = QLineEdit(s.sender_company)
        self.optout = QCheckBox("Incluir frase de não contato nas mensagens (recomendado pela LGPD)")
        self.optout.setChecked(s.include_optout)
        self.threshold = _spin(0, 100, s.qualified_threshold)
        form.addRow("Seu nome", self.sender); form.addRow("Sua empresa", self.company)
        form.addRow("Qualificada a partir de", self.threshold)
        lay.addLayout(form); lay.addWidget(self.optout)
        grid.addWidget(f, 0, 0, Qt.AlignTop)

        # Google
        g, gl = style.card("Google Places (pago, chave própria)")
        self.g_enabled = QCheckBox("Usar o Google Places nas pesquisas"); self.g_enabled.setChecked(s.google_enabled)
        gl.addWidget(self.g_enabled)
        self.g_key = QLineEdit(); self.g_key.setEchoMode(QLineEdit.Password)
        self.g_key.setPlaceholderText("Cole a chave de API (Places API New) e clique em Salvar chave")
        self.g_key_info = style.label("", "caption")
        gl.addLayout(style.row(self.g_key, style.button("Salvar chave", self._save_google_key, "outline"),
                               style.button("Remover", lambda: self._save_secret("google_places", ""), "ghost"),
                               stretch_at=0))
        gl.addWidget(self.g_key_info)
        gform = QFormLayout(); gform.setSpacing(8)
        self.g_price = QDoubleSpinBox(); self.g_price.setRange(0, 1000); self.g_price.setDecimals(2)
        self.g_price.setValue(s.google_price_per_1000_usd); self.g_price.setPrefix("US$ ")
        self.g_price.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.g_per_search = _spin(1, 2000, s.google_max_requests_per_search)
        self.g_month = _spin(0, 100000, s.google_monthly_request_cap)
        self.g_pages = _spin(1, 3, s.google_max_pages)
        gform.addRow("Preço por 1.000 requisições", self.g_price)
        gform.addRow("Máximo por pesquisa", self.g_per_search)
        gform.addRow("Máximo por mês", self.g_month)
        gform.addRow("Páginas por consulta (20 resultados cada)", self.g_pages)
        gl.addLayout(gform)
        self.g_usage = style.label("", "lfWarn", wrap=True); gl.addWidget(self.g_usage)
        gl.addWidget(style.label("Pelos termos do Google, nome, telefone e endereço vindos do Google ficam no máximo "
                                 "30 dias no seu computador e não são exportados. Confira a tabela de preços atual "
                                 "do Google Maps Platform e o crédito gratuito mensal da sua conta.", "caption",
                                 wrap=True))
        grid.addWidget(g, 1, 0)

        # OSM
        o, ol = style.card("OpenStreetMap (gratuito)")
        self.o_enabled = QCheckBox("Usar o OpenStreetMap (Overpass API)"); self.o_enabled.setChecked(s.osm_enabled)
        self.o_url = QLineEdit(s.overpass_url)
        ol.addWidget(self.o_enabled)
        ol.addLayout(style.row(style.label("Servidor", "lfFieldName"), self.o_url, stretch_at=1))
        ol.addWidget(style.label("Dados © colaboradores do OpenStreetMap (ODbL). Consultas espaçadas, uma por área, "
                                 "conforme a política de uso da Overpass.", "caption", wrap=True))
        grid.addWidget(o, 2, 0)

        # CNPJ
        c, cl = style.card("Base aberta de CNPJ (Receita Federal)")
        self.c_enabled = QCheckBox("Usar o índice local de CNPJ nas pesquisas"); self.c_enabled.setChecked(s.cnpj_enabled)
        cl.addWidget(self.c_enabled)
        cl.addWidget(style.label("1. Baixe na página oficial os arquivos Estabelecimentos*.zip, Empresas*.zip e "
                                 "Municipios.zip (vários GB). 2. Escolha a pasta e as UFs. 3. Importe: só empresas "
                                 "ativas do setor imobiliário são guardadas. E-mails e sócios não são importados.",
                                 "caption", wrap=True))
        self.c_folder = QLineEdit(); self.c_folder.setPlaceholderText("Pasta com os arquivos da Receita")
        cl.addLayout(style.row(self.c_folder, style.button("  Escolher", self._choose_folder, "ghost",
                                                           icons.icon("folder")), stretch_at=0))
        self.c_ufs = QLineEdit(); self.c_ufs.setPlaceholderText("UFs separadas por vírgula (ex.: SP, RJ). Vazio = Brasil")
        cl.addWidget(self.c_ufs)
        self.c_bar = QProgressBar(); self.c_bar.setRange(0, 1000); self.c_bar.setTextVisible(False); self.c_bar.hide()
        cl.addWidget(self.c_bar)
        self.c_info = style.label("", "caption", wrap=True); cl.addWidget(self.c_info)
        self.c_import = style.button("  Importar", self._import, "outline", icons.icon("download", theme.ACCENT_HI))
        self.c_cancel = style.button("Cancelar", self._cancel_import, "danger"); self.c_cancel.setEnabled(False)
        cl.addLayout(style.row(style.button("  Página oficial", lambda: self.ctx.open_url(OFFICIAL_PAGE), "ghost",
                                            icons.icon("globe")), None, self.c_import, self.c_cancel))
        grid.addWidget(c, 0, 1)

        # IA
        a, al = style.card("Mensagens por IA (opcional)")
        self.a_enabled = QCheckBox("Permitir variações de mensagem por IA (API da Anthropic)")
        self.a_enabled.setChecked(s.ai_enabled)
        al.addWidget(self.a_enabled)
        self.a_key = QLineEdit(); self.a_key.setEchoMode(QLineEdit.Password)
        self.a_key.setPlaceholderText("Chave da API (ou variável de ambiente ANTHROPIC_API_KEY)")
        al.addLayout(style.row(self.a_key, style.button("Salvar chave", self._save_ai_key, "outline"),
                               style.button("Remover", lambda: self._save_secret("anthropic", ""), "ghost"),
                               stretch_at=0))
        self.a_key_info = style.label("", "caption"); al.addWidget(self.a_key_info)
        aform = QFormLayout(); aform.setSpacing(8)
        self.a_model = QLineEdit(s.ai_model); self.a_cap = _spin(0, 1000, s.ai_daily_cap)
        aform.addRow("Modelo", self.a_model); aform.addRow("Máximo por dia", self.a_cap)
        al.addLayout(aform)
        al.addWidget(style.label("Só o nome, a cidade e o segmento da empresa vão para a IA, e só quando você clica "
                                 "em “Variação com IA”. Sem chave, os modelos determinísticos funcionam normalmente.",
                                 "caption", wrap=True))
        grid.addWidget(a, 1, 1)

        # LGPD
        p, pl = style.card("Dados e privacidade (LGPD)")
        pl.addWidget(style.label("Os contatos ficam só neste computador, em " + str(ctx.base) + ". A origem de cada "
                                 "dado (fonte e data) é registrada. Pedidos de não contato são respeitados em novas "
                                 "pesquisas.", "caption", wrap=True))
        self.p_info = style.label("", "body", wrap=True); pl.addWidget(self.p_info)
        pl.addLayout(style.row(style.button("  Aplicar prazos de guarda agora", self._purge, "ghost",
                                            icons.icon("refresh")),
                               style.button("  Limpar lista de não contato", self._clear_suppression, "ghost"),
                               None))
        pl.addLayout(style.row(None, style.button("  Apagar todos os dados do Lead Finder", self._delete_all, "danger",
                                                  icons.icon("trash", theme.DANGER))))
        grid.addWidget(p, 2, 1)

        self.btn_save = style.button("  Salvar configurações", self.save, "primary", icons.icon("check", theme.ON_ACCENT))
        self.saved = style.label("", "lfOk")
        grid.addLayout(style.row(None, self.saved, self.btn_save), 3, 0, 1, 2)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(4, 1)

    # --------------------------------------------------------------------------------- salvar
    def save(self) -> None:
        s = self.ctx.settings
        s.sender_name, s.sender_company = self.sender.text().strip(), self.company.text().strip() or "7x Intelligence"
        s.include_optout, s.qualified_threshold = self.optout.isChecked(), self.threshold.value()
        s.google_enabled = self.g_enabled.isChecked()
        s.google_price_per_1000_usd = float(self.g_price.value())
        s.google_max_requests_per_search, s.google_monthly_request_cap = self.g_per_search.value(), self.g_month.value()
        s.google_max_pages = self.g_pages.value()
        s.osm_enabled = self.o_enabled.isChecked()
        url = self.o_url.text().strip()
        if url.startswith("https://"):
            s.overpass_url = url
        s.cnpj_enabled = self.c_enabled.isChecked()
        s.ai_enabled, s.ai_model, s.ai_daily_cap = self.a_enabled.isChecked(), self.a_model.text().strip(), self.a_cap.value()
        try:
            self.ctx.save_settings()
        except OSError as e:
            QMessageBox.warning(self, "Configurações", f"Não foi possível salvar: {e}")
            return
        self.saved.setText("Configurações salvas.")
        self.refresh()

    def _save_secret(self, name: str, value: str) -> None:
        try:
            self.ctx.secrets.set(name, value)
        except OSError as e:
            QMessageBox.warning(self, "Chave de API", f"Não foi possível salvar: {e}")
            return
        self.g_key.clear(); self.a_key.clear()
        self.saved.setText("Chave salva." if value else "Chave removida.")
        self.ctx.settings_changed.emit()
        self.refresh()

    def _save_google_key(self) -> None:
        if self.g_key.text().strip():
            self._save_secret("google_places", self.g_key.text())

    def _save_ai_key(self) -> None:
        if self.a_key.text().strip():
            self._save_secret("anthropic", self.a_key.text())

    # ---------------------------------------------------------------------------------- CNPJ
    def _choose_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Pasta com os arquivos da Receita")
        if d:
            self.c_folder.setText(d)

    def parse_ufs(self) -> list[str] | None:
        raw = [u.strip().upper() for u in self.c_ufs.text().replace(";", ",").split(",") if u.strip()]
        bad = [u for u in raw if u not in UF_NAME]
        if bad:
            self.c_info.setText("UF inválida: " + ", ".join(bad))
            return None
        return raw

    def _import(self) -> None:
        folder = self.c_folder.text().strip()
        if not folder or not Path(folder).is_dir():
            self.c_info.setText("Escolha a pasta com os arquivos baixados da Receita.")
            return
        ufs = self.parse_ufs()
        if ufs is None:
            return
        index = self.ctx.base / INDEX_NAME
        self.c_bar.show(); self.c_bar.setValue(0)
        self.c_import.setEnabled(False); self.c_cancel.setEnabled(True)
        self.c_info.setText("Importando… pode levar vários minutos para o Brasil inteiro.")

        def done(rep):
            self._import_finished()
            missing = rep.kept - rep.with_legal_name if rep.kept else 0
            self.c_info.setText((f"Importação concluída: {rep.kept:,} empresa(s) ativa(s) do setor de "
                                 f"{rep.rows_read:,} linhas lidas."
                                 + (f" Atenção: {missing:,} sem razão social/natureza jurídica — inclua os arquivos "
                                    "Empresas*.zip; sem a natureza, ficam fora da busca padrão." if missing > 0 else ""))
                                .replace(",", "."))
            self.ctx.settings_changed.emit()
            self.refresh()
        self.import_task = self.ctx.run(lambda t: import_open_data(Path(folder), index, ufs, t, t.progress),
                                        name="lead-finder-cnpj-import",
                                        on_progress=lambda f, m: (self.c_bar.setValue(int(1000 * (f or 0))),
                                                                  self.c_info.setText(m)),
                                        on_done=done,
                                        on_error=lambda e: (self._import_finished(), self.c_info.setText(friendly_error(e))),
                                        on_cancel=lambda: (self._import_finished(),
                                                           self.c_info.setText("Importação cancelada; o índice anterior foi mantido.")))

    def _import_finished(self) -> None:
        self.c_bar.hide(); self.c_import.setEnabled(True); self.c_cancel.setEnabled(False)

    def _cancel_import(self) -> None:
        if self.import_task is not None:
            self.import_task.cancel()

    # ---------------------------------------------------------------------------------- LGPD
    def _purge(self) -> None:
        self.ctx.db.call(lambda s: s.purge_expired(), on_done=lambda n: (
            self.saved.setText(f"{n} registro(s) com prazo vencido tiveram os dados apagados."),
            self.ctx.leads_changed.emit(), self.refresh()))

    def _clear_suppression(self) -> None:
        box = QMessageBox(QMessageBox.Question, "Não contatar",
                          "Limpar a lista de não contato? Empresas que pediram para não ser contatadas poderão voltar "
                          "em novas pesquisas.", QMessageBox.Yes | QMessageBox.No, self)
        if box.exec() == QMessageBox.Yes:
            self.ctx.db.call(lambda s: s.clear_suppression(), on_done=lambda _r: self.refresh())

    def _delete_all(self) -> None:
        box = QMessageBox(QMessageBox.Warning, "Apagar dados",
                          "Apagar TODAS as empresas, históricos, notas e lembretes do Lead Finder deste computador? "
                          "A lista de não contato é mantida.", QMessageBox.Yes | QMessageBox.No, self)
        if box.exec() == QMessageBox.Yes:
            self.ctx.db.call(lambda s: s.delete_all(), on_done=lambda _r: (self.ctx.leads_changed.emit(),
                                                                           self.saved.setText("Dados apagados."),
                                                                           self.refresh()))

    # ------------------------------------------------------------------------------- atualizar
    def refresh(self) -> None:
        sec = self.ctx.secrets
        for name, label in (("google_places", self.g_key_info), ("anthropic", self.a_key_info)):
            origin = sec.origin(name)
            label.setText({"ambiente": "Chave definida por variável de ambiente.",
                           "arquivo": f"Chave salva neste computador ({mask(sec.get(name))})."}.get(origin,
                                                                                                   "Nenhuma chave."))
        base = self.ctx.base
        info_task = lambda s: (s.usage_month("google_places"), s.usage_today("anthropic"),  # noqa: E731
                               s.suppression_count(), s.facets()["total"], index_info(base / INDEX_NAME))
        self.ctx.db.call(info_task, on_done=self._show_info)

    def _show_info(self, res) -> None:
        (g_units, g_cost), (ai_today, _), supp, total, idx = res
        cap = self.ctx.settings.google_monthly_request_cap
        self.g_usage.setText(f"Uso neste mês: {g_units} de {cap} requisições (≈ US$ {g_cost:.2f}). "
                             f"IA hoje: {ai_today} de {self.ctx.settings.ai_daily_cap}.")
        self.p_info.setText(f"{total} empresa(s) no CRM · {supp} identificador(es) na lista de não contato.")
        if idx.get("rows") and not (self.import_task and self.import_task.running()):
            self.c_info.setText(f"Índice local: {idx['rows']:,} empresas · UFs {idx.get('ufs', '')} · importado em "
                                f"{idx.get('imported_at', '')[:10]}".replace(",", "."))

    def on_show(self) -> None:
        self.saved.setText("")
        self.refresh()
