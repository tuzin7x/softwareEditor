"""Aba Mensagens: criar, personalizar, editar, salvar modelos, copiar e abrir no WhatsApp."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QInputDialog, QLineEdit, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QWidget)

from mobilia.ui import theme
from mobilia.ui.design_widgets import Segmented

from .. import messages as msg
from ..models import DO_NOT_CONTACT
from ..tasks import friendly_error
from . import icons, style
from .dialogs import message_context, open_whatsapp


class MessagesView(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.custom: list[dict] = []
        self.lead: dict | None = None
        self.ai_task = None
        root = QGridLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setHorizontalSpacing(16)

        left, ll = style.card("Modelos")
        self.list = QListWidget(); self.list.setObjectName("lfList")
        self.list.currentItemChanged.connect(lambda *_: self._template_selected())
        ll.addWidget(self.list, 1)
        ll.addWidget(style.label("Marcadores: {empresa}, {cidade}, {meu_nome}, {minha_empresa}. Um trecho entre [[ ]] "
                                 "some quando um marcador dele está vazio.", "caption", wrap=True))
        self.btn_del = style.button("  Excluir modelo", self._delete_template, "ghost", icons.icon("trash"))
        ll.addWidget(self.btn_del)
        root.addWidget(left, 0, 0)

        right, rl = style.card("Mensagem", glow=True)
        self.lead_search = QLineEdit(); self.lead_search.setPlaceholderText("Buscar empresa do CRM…")
        self.lead_search.returnPressed.connect(self._find_leads)
        self.lead_combo = QComboBox(); self.lead_combo.setMinimumWidth(260)
        self.lead_combo.currentIndexChanged.connect(lambda *_: self._lead_chosen())
        rl.addLayout(style.row(style.label("PARA", "lfFieldName"), self.lead_search,
                               style.button("", self._find_leads, "ghost", icons.icon("search")), self.lead_combo,
                               stretch_at=1))
        self.company = QLineEdit(); self.company.setPlaceholderText("Nome da imobiliária (personalização)")
        self.city = QLineEdit(); self.city.setPlaceholderText("Cidade (opcional)")
        rl.addLayout(style.row(self.company, self.city, stretch_at=0))
        self.tone = Segmented(list(msg.TONES), "profissional")
        self.tone.setMinimumWidth(260)
        self.tone.changed.connect(lambda *_: self._tone_changed())
        self.optout = QCheckBox("Frase de não contato"); self.optout.setChecked(ctx.settings.include_optout)
        rl.addLayout(style.row(style.label("TOM", "lfFieldName"), self.tone, None, self.optout))
        self.editor = QPlainTextEdit(); self.editor.setMinimumHeight(220)
        self.editor.textChanged.connect(lambda: self.count.setText(f"{len(self.editor.toPlainText())} caracteres"))
        rl.addWidget(self.editor, 1)
        self.count = style.label("", "caption"); rl.addWidget(self.count)
        self.info = style.label("", "caption", wrap=True); rl.addWidget(self.info)
        self.btn_make = style.button("  Criar mensagem", self.generate, "outline", icons.icon("message", theme.ACCENT_HI))
        self.btn_ai = style.button("  Variação com IA", self.generate_ai, "ghost", icons.icon("sparkle"),
                                   "Opcional: usa a sua chave de API configurada. O texto sempre passa pela sua revisão.")
        self.btn_save = style.button("  Salvar como modelo", self._save_template, "ghost", icons.icon("note"))
        self.btn_copy = style.button("  Copiar", self._copy, "ghost", icons.icon("copy"))
        self.btn_wa = style.button("  OPEN WHATSAPP", self._open_wa, "lfWa", icons.icon("whatsapp", style.WA_COLOR))
        rl.addLayout(style.row(self.btn_make, self.btn_ai, self.btn_save, None, self.btn_copy, self.btn_wa))
        rl.addWidget(style.label("Nada é enviado automaticamente. O WhatsApp abre com o texto e você revisa e envia. "
                                 "A mensagem usa só dados reais do cadastro: nada é inventado sobre a empresa.",
                                 "hint", wrap=True))
        root.addWidget(right, 0, 1)
        root.setColumnStretch(0, 2); root.setColumnStretch(1, 5)
        ctx.compose_for.connect(self.compose_for)
        ctx.settings_changed.connect(self._update_buttons)
        self._fill_templates()
        self._update_buttons()

    # ------------------------------------------------------------------------------------ modelos
    def _fill_templates(self, select: str = "") -> None:
        cur = select or (self.list.currentItem().data(Qt.UserRole)["name"] if self.list.currentItem() else "")
        self.list.blockSignals(True); self.list.clear()
        for t in list(msg.BUILTIN_TEMPLATES) + self.custom:
            builtin = t in msg.BUILTIN_TEMPLATES
            it = QListWidgetItem(("" if builtin else "★ ") + t["name"])
            it.setData(Qt.UserRole, t)
            it.setToolTip("Modelo padrão" if builtin else "Modelo salvo por você")
            self.list.addItem(it)
        names = [self.list.item(i).data(Qt.UserRole)["name"] for i in range(self.list.count())]
        self.list.setCurrentRow(names.index(cur) if cur in names else 0)
        self.list.blockSignals(False)
        # só um modelo diferente (ou o primeiro) gera texto novo: recarregar a lista não apaga o rascunho editado
        if not cur or cur not in names or select:
            self._template_selected()
        else:
            t = self.current_template()
            self.btn_del.setEnabled(bool(t) and t not in msg.BUILTIN_TEMPLATES)

    def load_templates(self) -> None:
        self.ctx.db.call(lambda s: s.templates(), on_done=self._templates_loaded)

    def _templates_loaded(self, rows) -> None:
        self._templates_ready = True       # carregado, mesmo que a pessoa não tenha modelos próprios
        self.custom = [dict(r) for r in rows]
        self._fill_templates()

    def current_template(self) -> dict | None:
        it = self.list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _template_selected(self) -> None:
        t = self.current_template()
        self.btn_del.setEnabled(bool(t) and t not in msg.BUILTIN_TEMPLATES)
        if t:
            self.tone.set_value(t.get("tone", "profissional"))
            self.generate()

    def _tone_changed(self) -> None:
        t = self.current_template()
        if t and t.get("tone") != self.tone.value():
            for i in range(self.list.count()):
                cand = self.list.item(i).data(Qt.UserRole)
                if cand.get("tone") == self.tone.value() and cand["name"].split(" · ")[0] == t["name"].split(" · ")[0]:
                    self.list.setCurrentRow(i)
                    return
        self.generate()

    def _save_template(self) -> None:
        body = self.editor.toPlainText().strip()
        try:
            msg.validate(body)
        except ValueError as e:
            self.info.setText(str(e)); return
        name, ok = QInputDialog.getText(self, "Salvar modelo", "Nome do modelo:")
        if not ok or not name.strip():
            return
        tone = self.tone.value()
        self.ctx.db.call(lambda s: s.save_template(name, tone, body),
                         on_done=lambda _r: (self.info.setText(f"Modelo “{name.strip()}” salvo."),
                                             self._reload_and_select(name.strip())),
                         on_error=lambda e: self.info.setText(friendly_error(e)))

    def _reload_and_select(self, name: str) -> None:
        self.ctx.db.call(lambda s: s.templates(), on_done=lambda rows: (setattr(self, "custom", [dict(r) for r in rows]),
                                                                        self._fill_templates(name)))

    def _delete_template(self) -> None:
        t = self.current_template()
        if not t or t in msg.BUILTIN_TEMPLATES:
            return
        self.ctx.db.call(lambda s: s.delete_template(t["name"]), on_done=lambda _r: self.load_templates())

    # ------------------------------------------------------------------------------------ empresa
    def _find_leads(self) -> None:
        text = self.lead_search.text().strip()
        self.ctx.db.call(lambda s: s.page({"text": text, "hide_dnc": True}, "name", False, 0, 50)[0],
                         on_done=self._leads_found)

    def _leads_found(self, rows) -> None:
        # a lista nova substitui a anterior: o destinatário ativo deixa de valer até a pessoa escolher outro
        self._clear_recipient()
        self.lead_combo.blockSignals(True); self.lead_combo.clear()
        self.lead_combo.addItem("— escolha uma empresa —", None)
        for r in rows:
            self.lead_combo.addItem(f"{r['name']} · {r['city']}/{r['uf']}".rstrip(" ·/"), r["id"])
        self.lead_combo.blockSignals(False)
        if self.lead_combo.count() == 2:
            self.lead_combo.setCurrentIndex(1)
        self.info.setText(f"{len(rows)} empresa(s) encontrada(s)." if rows else "Nenhuma empresa encontrada.")

    def _lead_chosen(self) -> None:
        lid = self.lead_combo.currentData()
        self._clear_recipient()
        if lid:
            # a resposta do banco é assíncrona: só vale se o seletor ainda mostrar a mesma empresa
            self.ctx.db.call(lambda s: s.get(lid),
                             on_done=lambda lead: self._set_lead(lead) if self.lead_combo.currentData() == lid else None)
        else:
            self.generate()

    def _clear_recipient(self) -> None:
        """Esquece o destinatário (o botão do WhatsApp fica desligado). Campos preenchidos a partir dele são
        limpos; um nome digitado à mão pela pessoa é mantido."""
        old = self.lead
        self.lead = None
        if old:
            prev = message_context(old, self.ctx.settings)
            if self.company.text().strip() == prev.empresa:
                self.company.clear()
            if self.city.text().strip() == prev.cidade:
                self.city.clear()
        self._update_buttons()

    def _set_lead(self, lead: dict | None) -> None:
        self.lead = lead
        if lead:
            ctx = message_context(lead, self.ctx.settings)
            self.company.setText(ctx.empresa); self.city.setText(ctx.cidade)
        self._update_buttons()
        self.generate()

    def compose_for(self, lead_id: int) -> None:
        self._composing = True          # a busca automática ao abrir a aba não pode trocar esta empresa

        def got(lead):
            self._composing = False
            if lead is None:
                return
            self.lead_combo.blockSignals(True); self.lead_combo.clear()
            self.lead_combo.addItem(f"{lead['name']} · {lead['city']}/{lead['uf']}", lead["id"])
            self.lead_combo.blockSignals(False)
            self._set_lead(lead)
        self.ctx.db.call(lambda s: s.get(lead_id), on_done=got)

    # ------------------------------------------------------------------------------------ texto
    def context(self) -> msg.Context:
        s = self.ctx.settings
        return msg.Context(empresa=self.company.text().strip(), cidade=self.city.text().strip(),
                           meu_nome=s.sender_name, minha_empresa=s.sender_company or "7x Intelligence")

    def generate(self) -> None:
        t = self.current_template()
        if not t:
            return
        try:
            self.editor.setPlainText(msg.render(t["body"], self.context(), self.optout.isChecked()))
            self.info.setText("")
        except ValueError as e:
            self.info.setText(str(e))

    def generate_ai(self) -> None:
        s = self.ctx.settings
        key = self.ctx.secrets.get("anthropic")
        if not (s.ai_enabled and key):
            self.info.setText("A geração por IA está desativada ou sem chave. Configure em Configurações › IA.")
            return
        base, tone, mctx = self.editor.toPlainText(), self.tone.value(), self.context()
        category = (self.lead or {}).get("category", "")
        http = self.ctx.http

        def check_and_run(used):
            if used >= s.ai_daily_cap:
                self.info.setText(f"Limite diário de {s.ai_daily_cap} gerações por IA atingido.")
                return
            self.btn_ai.setEnabled(False)
            self.info.setText("Gerando uma variação com IA…")
            self.ai_task = self.ctx.run(
                lambda t: msg.generate_with_ai(http, key, s.ai_model, mctx, tone, base, category, t),
                name="lead-finder-ai", on_done=self._ai_done, on_error=self._ai_failed,
                on_cancel=lambda: self.btn_ai.setEnabled(True))
        self.ctx.db.call(lambda st: st.usage_today("anthropic")[0], on_done=check_and_run)

    def _ai_done(self, text: str) -> None:
        self.btn_ai.setEnabled(True)
        self.editor.setPlainText(text)
        self.info.setText("Variação criada pela IA. Revise antes de usar: confira se nada foi inventado.")
        self.ctx.db.call(lambda s: s.record_usage("anthropic", 1, 0.0))

    def _ai_failed(self, e) -> None:
        self.btn_ai.setEnabled(True)
        self.info.setText(friendly_error(e))

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.editor.toPlainText())
        self.ctx.status("Mensagem copiada.")
        if self.lead:
            lid = self.lead["id"]
            self.ctx.db.call(lambda s: s.add_event(lid, "mensagem_copiada", self.current_template()["name"]))

    def _open_wa(self) -> None:
        if not self.lead or self.lead_combo.currentData() != self.lead["id"]:
            self.info.setText("Escolha uma empresa do CRM para abrir o WhatsApp.")
            return
        open_whatsapp(self.ctx, self, self.lead["id"], self.editor.toPlainText())

    def _update_buttons(self) -> None:
        lead = self.lead
        self.btn_wa.setEnabled(bool(lead and lead.get("wa_number") and lead.get("status") != DO_NOT_CONTACT))
        self.btn_ai.setVisible(bool(self.ctx.settings.ai_enabled))

    def on_show(self) -> None:
        if not getattr(self, "_templates_ready", False):
            self.load_templates()
        if self.lead_combo.count() == 0 and not getattr(self, "_composing", False):
            self._find_leads()
