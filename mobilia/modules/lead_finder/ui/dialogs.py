"""Diálogos: revisar e abrir o WhatsApp, vincular projeto do AI Studio e cadastrar empresa manualmente."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QStackedWidget, QVBoxLayout, QWidget)

from mobilia.ui import theme

from .. import messages as msg
from .. import phone as ph
from .. import studio
from ..geo import UFS, UF_NAME
from ..models import DO_NOT_CONTACT, WA_CONFIRMED, WA_LABEL, Lead
from ..tasks import friendly_error
from . import icons, style


def message_context(lead: dict, settings) -> msg.Context:
    """Só dados reais do cadastro: nome e cidade da empresa e a identidade de quem envia."""
    return msg.Context(empresa=lead.get("name", "") if not lead.get("needs_refresh") else "",
                       cidade=lead.get("city", ""), meu_nome=settings.sender_name,
                       minha_empresa=settings.sender_company or "7x Intelligence")


class WhatsAppDialog(QDialog):
    """Revisão obrigatória antes de abrir o WhatsApp; depois, a pessoa confirma se enviou de fato."""

    def __init__(self, ctx, lead: dict, custom_templates: list[dict], parent=None, text: str = ""):
        super().__init__(parent)
        self.ctx, self.lead = ctx, lead
        self.custom = custom_templates
        self.opened = False
        self.sent_confirmed = False
        self.setWindowTitle("Abrir conversa no WhatsApp")
        self.setMinimumWidth(560)
        self.setStyleSheet(style.QSS)
        self.pages = QStackedWidget()
        lay = QVBoxLayout(self); lay.setContentsMargins(22, 20, 22, 18); lay.addWidget(self.pages)
        self.pages.addWidget(self._review_page(text))
        self.pages.addWidget(self._confirm_page())

    def _review_page(self, text: str) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(10)
        info = ph.normalize(self.lead.get("wa_number", ""))
        title = style.label(self.lead.get("name", ""), "title")
        lay.addWidget(title)
        num = style.label(f"WhatsApp {info.national}  ·  {WA_LABEL.get(self.lead.get('wa_status'), '')}", "subtitle")
        lay.addWidget(num)
        if self.lead.get("wa_status") != WA_CONFIRMED:
            lay.addWidget(style.label("Este número ainda não foi confirmado como WhatsApp. Se a conversa não abrir, "
                                      "o número pode não ter WhatsApp — registre isso no CRM.", "lfWarn", wrap=True))
        self.template = QComboBox()
        for t in list(msg.BUILTIN_TEMPLATES) + list(self.custom):
            self.template.addItem(t["name"], t)
        self.optout = QCheckBox("Incluir frase de não contato (recomendado)")
        self.optout.setChecked(bool(self.ctx.settings.include_optout))
        lay.addLayout(style.row(style.label("Modelo", "lfFieldName"), self.template, None, self.optout, stretch_at=1))
        self.editor = QPlainTextEdit()
        self.editor.setMinimumHeight(170)
        lay.addWidget(self.editor)
        self.counter = style.label("", "caption")
        lay.addWidget(self.counter)
        self.error = style.label("", "lfDanger", wrap=True); self.error.hide()
        lay.addWidget(self.error)
        lay.addWidget(style.label("O WhatsApp abre com o texto preenchido. Revise e envie por lá: nada é enviado "
                                  "automaticamente.", "hint", wrap=True))
        copy = style.button("  Copiar", self._copy, icon=icons.icon("copy"))
        cancel = style.button("Cancelar", self.reject, "ghost")
        self.btn_open = style.button("  Abrir no WhatsApp", self._open, "lfWa", icons.icon("whatsapp", style.WA_COLOR))
        self.btn_open.setMinimumHeight(40)
        lay.addLayout(style.row(copy, None, cancel, self.btn_open))
        self.template.currentIndexChanged.connect(lambda *_: self._render())
        self.optout.toggled.connect(lambda *_: self._render())
        self.editor.textChanged.connect(self._count)
        if text:
            self.editor.setPlainText(text)
        else:
            self._render()
        return w

    def _confirm_page(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        lay.addWidget(style.label("WhatsApp aberto", "title"))
        lay.addWidget(style.label("A conversa foi aberta com a mensagem preenchida. Abrir o WhatsApp não significa "
                                  "que a mensagem foi enviada.\n\nVocê enviou a mensagem?", "body", wrap=True))
        lay.addStretch(1)
        no = style.button("Ainda não", self.reject, "ghost")
        yes = style.button("  Sim, enviei", self._confirm_sent, "primary", icons.icon("check", theme.ON_ACCENT))
        lay.addLayout(style.row(None, no, yes))
        return w

    def _render(self) -> None:
        t = self.template.currentData()
        if not t:
            return
        try:
            self.editor.setPlainText(msg.render(t["body"], message_context(self.lead, self.ctx.settings),
                                                self.optout.isChecked()))
        except ValueError as e:
            self._fail(str(e))

    def _count(self) -> None:
        n = len(self.editor.toPlainText())
        self.counter.setText(f"{n} caracteres")

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.editor.toPlainText())
        self.ctx.status("Mensagem copiada.")

    def _fail(self, text: str) -> None:
        self.error.setText(text); self.error.setVisible(bool(text))

    def _open(self) -> None:
        if self.lead.get("status") == DO_NOT_CONTACT:
            return self._fail("Esta empresa pediu para não ser contatada.")
        text = self.editor.toPlainText().strip()
        if not text:
            return self._fail("Escreva a mensagem antes de abrir o WhatsApp.")
        try:
            url = ph.wa_me_url(self.lead.get("wa_number", ""), text)
        except ValueError as e:
            return self._fail(str(e))
        # verificação final no banco, sem travar a interface: a lista de não contato pode ter mudado
        self.btn_open.setEnabled(False)
        number = self.lead.get("wa_number", "")
        self.ctx.db.call(lambda s: s.phone_blocked(number), on_done=lambda blocked: self._open_checked(url, text, blocked),
                         on_error=lambda e: (self.btn_open.setEnabled(True), self._fail(friendly_error(e))))

    def _open_checked(self, url: str, text: str, blocked: bool) -> None:
        self.btn_open.setEnabled(True)
        if blocked:
            return self._fail("Este número está na lista de não contato.")
        self.ctx.open_url(url)
        self.opened = True
        self.ctx.db.call(lambda s: s.add_event(self.lead["id"], "whatsapp_aberto", self.template.currentText(),
                                               {"caracteres": len(text)}),
                         on_done=lambda _r: self.ctx.leads_changed.emit())
        self.pages.setCurrentIndex(1)

    def _confirm_sent(self) -> None:
        def done(_r):
            self.sent_confirmed = True
            self.ctx.leads_changed.emit()
            self.ctx.status("Mensagem registrada como enviada.")
            self.accept()
        self.ctx.db.call(lambda s: s.add_event(self.lead["id"], "mensagem_enviada", self.template.currentText()),
                         on_done=done, on_error=lambda e: self.ctx.status(friendly_error(e)))


class StudioPickerDialog(QDialog):
    """Lista os projetos do AI Studio (só leitura) para vincular um deles à empresa."""

    THUMB = QSize(96, 64)

    def __init__(self, ctx, lead_id: int, parent=None, base: Path | None = None):
        super().__init__(parent)
        self.ctx, self.lead_id, self.base = ctx, lead_id, base
        self.projects: list[studio.StudioProject] = []
        self.setWindowTitle("Vincular projeto do AI Studio")
        self.setMinimumSize(620, 460)
        self.setStyleSheet(style.QSS)
        lay = QVBoxLayout(self); lay.setContentsMargins(22, 20, 22, 18); lay.setSpacing(10)
        lay.addWidget(style.label("Projetos e vídeos do AI Studio", "title"))
        lay.addWidget(style.label("Escolha uma geração para associar a esta empresa (ex.: a demonstração enviada). "
                                  "Os arquivos não são alterados nem enviados.", "hint", wrap=True))
        self.list = QListWidget(); self.list.setObjectName("lfList"); self.list.setIconSize(self.THUMB)
        lay.addWidget(self.list, 1)
        self.what = QComboBox()
        self.empty = style.label("Carregando projetos…", "muted"); lay.addWidget(self.empty)
        self.btn_folder = style.button("  Abrir pasta", self._open_folder, icon=icons.icon("folder"))
        self.btn_link = style.button("  Vincular", self._link, "primary", icons.icon("link", theme.ON_ACCENT))
        lay.addLayout(style.row(style.label("Vincular", "lfFieldName"), self.what, self.btn_folder, None,
                                style.button("Cancelar", self.reject, "ghost"), self.btn_link, stretch_at=1))
        self.list.currentRowChanged.connect(self._changed)
        self.btn_link.setEnabled(False)
        ctx.run(self._load, name="lead-finder-studio", on_done=self._loaded,
                on_error=lambda e: self.empty.setText(friendly_error(e)))

    def _load(self, task):
        projects = studio.list_projects(self.base)
        thumbs = []
        for p in projects:
            img = QImage(p.thumb) if p.thumb else QImage()
            thumbs.append(img.scaled(self.THUMB * 2, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                          if not img.isNull() else img)
        return projects, thumbs

    def _loaded(self, res) -> None:
        self.projects, thumbs = res
        self.empty.setText("" if self.projects else "Nenhum projeto do AI Studio encontrado neste computador.")
        self.empty.setVisible(not self.projects)
        for p, img in zip(self.projects, thumbs):
            it = QListWidgetItem(f"{p.title}\n{p.created} · {len(p.images)} imagem(ns)"
                                 + (f" · {len(p.videos)} vídeo(s)" if p.videos else ""))
            if not img.isNull():
                pm = QPixmap.fromImage(img); pm.setDevicePixelRatio(2.0)
                it.setIcon(QIcon(pm))
            self.list.addItem(it)
        if self.projects:
            self.list.setCurrentRow(0)

    def _changed(self, row: int) -> None:
        self.what.clear()
        ok = 0 <= row < len(self.projects)
        self.btn_link.setEnabled(ok)
        if not ok:
            return
        p = self.projects[row]
        self.what.addItem("Projeto inteiro", ("projeto", p.path, p.title))
        for v in p.videos:
            self.what.addItem(f"Vídeo {Path(v).name}", ("video", v, f"{p.title} · vídeo"))
        for im in p.images[:8]:
            self.what.addItem(f"Imagem {Path(im).name}", ("imagem", im, f"{p.title} · imagem"))

    def _open_folder(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.projects):
            self.ctx.open_url(QUrl.fromLocalFile(self.projects[row].path).toString())

    def _link(self) -> None:
        data = self.what.currentData()
        if not data:
            return
        kind, path, title = data
        self.ctx.db.call(lambda s: s.link_media(self.lead_id, path, kind, title),
                         on_done=lambda _r: (self.ctx.leads_changed.emit(), self.accept()),
                         on_error=lambda e: self.empty.setText(friendly_error(e)))


class ManualLeadDialog(QDialog):
    """Cadastro manual (ex.: indicação). Fonte ``manual``: o próprio usuário é a origem do dado."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nova empresa")
        self.setMinimumWidth(460)
        self.setStyleSheet(style.QSS)
        lay = QVBoxLayout(self); lay.setContentsMargins(22, 20, 22, 18)
        lay.addWidget(style.label("Cadastrar empresa", "title"))
        form = QFormLayout(); form.setSpacing(8)
        self.f = {k: QLineEdit() for k in ("name", "city", "phone", "website", "cnpj", "category")}
        self.uf = QComboBox(); self.uf.addItem("—", "")
        for u in UFS:
            self.uf.addItem(f"{u} · {UF_NAME[u]}", u)
        labels = {"name": "Nome *", "city": "Cidade", "phone": "Telefone", "website": "Site", "cnpj": "CNPJ",
                  "category": "Categoria"}
        for k in ("name", "city"):
            form.addRow(labels[k], self.f[k])
        form.addRow("UF", self.uf)
        for k in ("phone", "website", "cnpj", "category"):
            form.addRow(labels[k], self.f[k])
        lay.addLayout(form)
        self.err = style.label("", "lfDanger"); lay.addWidget(self.err)
        lay.addLayout(style.row(None, style.button("Cancelar", self.reject, "ghost"),
                                style.button("Salvar", self._ok, "primary")))

    def _ok(self) -> None:
        if len(self.f["name"].text().strip()) < 2:
            self.err.setText("Informe o nome da empresa.")
            return
        self.accept()

    def lead(self) -> Lead:
        v = {k: w.text().strip() for k, w in self.f.items()}
        return Lead(source="manual", source_ref="", name=v["name"], city=v["city"], uf=self.uf.currentData() or "",
                    phone=v["phone"], website=v["website"], cnpj="".join(c for c in v["cnpj"] if c.isdigit()),
                    category=v["category"] or "Imobiliária", query="cadastro manual")




def open_whatsapp(ctx, parent, lead_id: int, text: str = "") -> None:
    """Busca a empresa e os modelos no banco e abre o diálogo de revisão (sem bloquear a interface)."""
    def show(res):
        lead, custom, blocked = res
        if lead is None:
            return
        if blocked:
            ctx.status("Este número está na lista de não contato.")
            return
        if lead["status"] == DO_NOT_CONTACT:
            ctx.status("Esta empresa pediu para não ser contatada.")
            return
        if not lead["wa_number"]:
            ctx.status("Esta empresa não tem número de WhatsApp conhecido.")
            return
        dlg = WhatsAppDialog(ctx, lead, custom, parent, text)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        ctx.last_dialog = dlg
        dlg.open()
    def fetch(s):
        lead = s.get(lead_id)
        return lead, s.templates(), bool(lead and lead["wa_number"] and s.phone_blocked(lead["wa_number"]))
    ctx.db.call(fetch, on_done=show,
                on_error=lambda e: ctx.status(friendly_error(e)))
