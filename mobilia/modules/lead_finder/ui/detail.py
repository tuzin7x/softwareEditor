"""Painel da empresa (CRM): dados com a origem de cada um, status, ações, qualificação, notas, lembretes,
vínculos com o AI Studio e histórico."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from PySide6.QtCore import QDateTime, Qt, QUrl, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QComboBox, QDateTimeEdit, QGridLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QScrollArea, QVBoxLayout, QWidget)

from mobilia.ui import theme

from .. import phone as ph
from ..models import (DO_NOT_CONTACT, EVENT_LABEL, PRIORITY_LABEL, STATUS_LABEL, STATUSES, WA_CONFIRMED, WA_LABEL,
                      WA_NONE)
from ..sources import TITLES
from ..sources.brasilapi import format_cnpj
from ..tasks import friendly_error
from . import icons, style
from .parts import ScoreBar
from .table import short_date


def local_dt(iso: str) -> str:
    try:
        d = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso or ""
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone().strftime("%d/%m/%Y %H:%M")


class LeadDetail(QWidget):
    whatsapp_requested = Signal(int)
    site_requested = Signal(int)
    cnpj_requested = Signal(int)
    google_refresh_requested = Signal(int)
    studio_requested = Signal(int)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.lead: dict | None = None
        self._loading = False
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setObjectName("lfScroll"); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget(); scroll.setWidget(body)
        self.lay = QVBoxLayout(body); self.lay.setContentsMargins(0, 0, 8, 0); self.lay.setSpacing(12)
        self.empty = style.label("Selecione uma empresa na tabela para ver os detalhes, registrar contatos e "
                                 "acompanhar o histórico.", "hint", wrap=True)
        self.lay.addWidget(self.empty)
        self.content = QWidget(); c = QVBoxLayout(self.content); c.setContentsMargins(0, 0, 0, 0); c.setSpacing(12)
        self.lay.addWidget(self.content); self.lay.addStretch(1)
        self.content.hide()

        # cabeçalho
        head, hl = style.card(glow=True)
        head.setObjectName("lfGlassHero")
        self.title = style.label("", "lfHeroTitle", wrap=True)
        self.subtitle = style.label("", "subtitle", wrap=True)
        self.badges = QLabel(); self.badges.setTextFormat(Qt.RichText)
        hl.addWidget(self.title); hl.addWidget(self.subtitle); hl.addWidget(self.badges)
        self.btn_wa = style.button("  OPEN WHATSAPP", lambda: self.whatsapp_requested.emit(self.lead["id"]), "lfWa",
                                   icons.icon("whatsapp", style.WA_COLOR))
        self.btn_wa.setMinimumHeight(38)
        self.btn_msg = style.button("  Mensagem", lambda: self.ctx.compose_for.emit(self.lead["id"]), "outline",
                                    icons.icon("message", theme.ACCENT_HI))
        self.btn_copy = style.button("", self._copy_phone, "ghost", icons.icon("copy"), "Copiar telefone")
        hl.addLayout(style.row(self.btn_wa, self.btn_msg, self.btn_copy, None))
        self.wa_note = style.label("", "caption", wrap=True); hl.addWidget(self.wa_note)
        c.addWidget(head)

        # CRM
        crm, cl = style.card("Prospecção")
        self.status = QComboBox()
        for k, label in STATUSES:
            self.status.addItem(label, k)
        self.status.currentIndexChanged.connect(self._status_changed)
        self.priority = QComboBox()
        for k in (0, 1, 2, 3):
            self.priority.addItem("Automática" if k == 0 else PRIORITY_LABEL[k], k)
        self.priority.currentIndexChanged.connect(self._priority_changed)
        self.wa_state = QComboBox()
        for k, label in (("", "WhatsApp: automático"), (WA_CONFIRMED, "WhatsApp confirmado"),
                         (WA_NONE, "Não tem WhatsApp")):
            self.wa_state.addItem(label, k)
        self.wa_state.currentIndexChanged.connect(self._wa_changed)
        g = QGridLayout(); g.setSpacing(8)
        g.addWidget(style.label("STATUS", "lfFieldName"), 0, 0); g.addWidget(self.status, 0, 1)
        g.addWidget(style.label("PRIORIDADE", "lfFieldName"), 1, 0); g.addWidget(self.priority, 1, 1)
        g.addWidget(style.label("WHATSAPP", "lfFieldName"), 2, 0); g.addWidget(self.wa_state, 2, 1)
        g.setColumnStretch(1, 1)
        cl.addLayout(g)
        cl.addWidget(style.label("Registrar o que aconteceu de fato:", "caption"))
        self.btn_sent = style.button("Mensagem enviada", lambda: self._event("mensagem_enviada"), small=True)
        self.btn_reply = style.button("Resposta recebida", lambda: self._event("resposta_recebida"), small=True)
        self.btn_demo = style.button("Demonstração enviada", lambda: self._event("demo_enviada"), small=True)
        ev = QGridLayout(); ev.setSpacing(6)
        ev.addWidget(self.btn_sent, 0, 0); ev.addWidget(self.btn_reply, 0, 1); ev.addWidget(self.btn_demo, 1, 0, 1, 2)
        cl.addLayout(ev)
        c.addWidget(crm)

        # dados
        data, self.data_lay = style.card("Dados da empresa")
        self.facts = QGridLayout(); self.facts.setHorizontalSpacing(12); self.facts.setVerticalSpacing(6)
        self.data_lay.addLayout(self.facts)
        self.sources_label = style.label("", "lfSource", wrap=True); self.data_lay.addWidget(self.sources_label)
        self.btn_cnpj = style.button("  Consultar CNPJ", lambda: self.cnpj_requested.emit(self.lead["id"]),
                                     icon=icons.icon("building"), small=True)
        self.btn_google = style.button("  Atualizar dados do Google",
                                       lambda: self.google_refresh_requested.emit(self.lead["id"]),
                                       icon=icons.icon("refresh"), small=True)
        self.data_lay.addLayout(style.row(self.btn_cnpj, self.btn_google, None))
        c.addWidget(data)

        # qualificação
        q, ql = style.card("Qualificação comercial")
        self.score_label = style.label("", "lfBig"); self.score_bar = ScoreBar()
        ql.addLayout(style.row(self.score_label, None)); ql.addWidget(self.score_bar)
        self.criteria = QLabel(); self.criteria.setTextFormat(Qt.RichText); self.criteria.setWordWrap(True)
        ql.addWidget(self.criteria)
        self.btn_site = style.button("  Analisar site", lambda: self.site_requested.emit(self.lead["id"]),
                                     icon=icons.icon("globe"), small=True,
                                     tip="Lê só a página principal do site oficial, se o robots.txt permitir.")
        self.btn_rescore = style.button("  Recalcular", self._rescore, icon=icons.icon("refresh"), small=True)
        ql.addLayout(style.row(self.btn_site, self.btn_rescore, None))
        c.addWidget(q)

        # notas
        n, nl = style.card("Notas comerciais")
        self.note_edit = QPlainTextEdit(); self.note_edit.setPlaceholderText("Ex.: falar com o gerente na sexta.")
        self.note_edit.setMaximumHeight(80)
        nl.addWidget(self.note_edit)
        nl.addLayout(style.row(None, style.button("  Adicionar nota", self._add_note, "outline",
                                                  icons.icon("note", theme.ACCENT_HI))))
        self.notes = QListWidget(); self.notes.setObjectName("lfList"); self.notes.setMaximumHeight(160)
        nl.addWidget(self.notes)
        c.addWidget(n)

        # lembretes
        r, rl = style.card("Acompanhamento")
        self.rem_when = QDateTimeEdit(); self.rem_when.setCalendarPopup(True)
        self.rem_when.setDisplayFormat("dd/MM/yyyy HH:mm")
        self.rem_text = QLineEdit(); self.rem_text.setPlaceholderText("O que fazer (ex.: enviar vídeo de exemplo)")
        rl.addLayout(style.row(self.rem_when, self.rem_text, style.button("", self._add_reminder, "outline",
                                                                          icons.icon("bell", theme.ACCENT_HI),
                                                                          "Criar lembrete"), stretch_at=1))
        self.reminders = QListWidget(); self.reminders.setObjectName("lfList"); self.reminders.setMaximumHeight(130)
        self.reminders.itemDoubleClicked.connect(self._complete_reminder)
        rl.addWidget(self.reminders)
        rl.addWidget(style.label("Duplo clique conclui um lembrete.", "caption"))
        c.addWidget(r)

        # AI Studio
        s, sl = style.card("AI Studio")
        self.media = QListWidget(); self.media.setObjectName("lfList"); self.media.setMaximumHeight(120)
        self.media.itemDoubleClicked.connect(self._open_media)
        sl.addWidget(self.media)
        self.btn_unlink = style.button("Remover vínculo", self._unlink, "ghost", small=True)
        sl.addLayout(style.row(style.button("  Vincular projeto ou vídeo",
                                            lambda: self.studio_requested.emit(self.lead["id"]), "outline",
                                            icons.icon("video", theme.ACCENT_HI)), self.btn_unlink, None))
        c.addWidget(s)

        # histórico
        h, hl2 = style.card("Histórico")
        self.history = QListWidget(); self.history.setObjectName("lfList"); self.history.setMinimumHeight(160)
        hl2.addWidget(self.history)
        c.addWidget(h)

        # privacidade
        d, dl = style.card("Privacidade (LGPD)")
        dl.addWidget(style.label("“Não contatar” bloqueia WhatsApp e lembretes e impede que a empresa volte em novas "
                                 "pesquisas. Excluir apaga a empresa e todo o histórico deste computador.", "caption",
                                 wrap=True))
        self.btn_dnc = style.button("  Não contatar", self._toggle_dnc, "danger", icons.icon("block", theme.DANGER))
        self.btn_delete = style.button("  Excluir dados", self._delete, "danger", icons.icon("trash", theme.DANGER))
        dl.addLayout(style.row(self.btn_dnc, self.btn_delete, None))
        c.addWidget(d)

    # ------------------------------------------------------------------------------------- dados
    def load(self, lead_id: int | None) -> None:
        if not lead_id:
            self.lead = None
            self.content.hide(); self.empty.show()
            return
        self.ctx.db.call(lambda s: s.get(lead_id), on_done=self._show)

    def reload(self) -> None:
        if self.lead:
            self.load(self.lead["id"])

    def _show(self, d: dict | None) -> None:
        if d is None:
            return self.load(None)
        self._loading = True
        self.lead = d
        self.empty.hide(); self.content.show()
        self.title.setText(d["name"])
        self.subtitle.setText(" · ".join(x for x in (d["legal_name"] if d["legal_name"] != d["name"] else "",
                                                     d["category"], " / ".join(x for x in (d["city"], d["uf"]) if x))
                                         if x))
        status_col = style.STATUS_COLOR.get(d["status"], theme.MUTED)
        badges = [style.pill_html(STATUS_LABEL.get(d["status"], d["status"]), status_col)]
        prio = max(d["priority"] or 0, d["priority_user"] or 0)
        if prio:
            badges.append(style.pill_html(f"Prioridade {PRIORITY_LABEL[prio].lower()}", style.PRIORITY_COLOR[prio]))
        if d["restricted"]:
            badges.append(style.pill_html("Google · 30 dias", theme.MUTED))
        self.badges.setText("&nbsp;&nbsp;".join(badges))
        dnc = d["status"] == DO_NOT_CONTACT
        self.btn_wa.setEnabled(bool(d["wa_number"]) and not dnc)
        if dnc:
            self.wa_note.setText("Esta empresa pediu para não ser contatada.")
        elif d["wa_number"]:
            info = ph.normalize(d["wa_number"])
            self.wa_note.setText(f"{info.national} — {WA_LABEL.get(d['wa_status'], '')}"
                                 + ("" if d["wa_status"] == WA_CONFIRMED else
                                    ". Celular não garante WhatsApp: confirme ao conversar."))
        else:
            self.wa_note.setText("Sem número de WhatsApp conhecido.")
        for b in (self.btn_msg, self.btn_sent, self.btn_demo):
            b.setEnabled(not dnc)
        self.status.setCurrentIndex(self.status.findData(d["status"]))
        self.priority.setCurrentIndex(self.priority.findData(d["priority_user"] or 0))
        self.wa_state.setCurrentIndex(max(0, self.wa_state.findData(d["wa_user"] or "")))
        self._facts(d)
        self._score(d)
        self.notes.clear()
        for n in d["notes"]:
            self.notes.addItem(f"{local_dt(n['at'])}\n{n['text']}")
        self.reminders.clear()
        for r in d["reminders"]:
            it = QListWidgetItem(("✓ " if r["done"] else "◷ ") + f"{local_dt(r['due_at'])}  {r['text']}")
            it.setData(Qt.UserRole, r["id"])
            if r["done"]:
                it.setForeground(Qt.gray)
            self.reminders.addItem(it)
        self.rem_when.setDateTime(QDateTime.currentDateTime().addDays(3))
        self.media.clear()
        for m in d["media"]:
            it = QListWidgetItem(icons.icon("video" if m["kind"] == "video" else "folder"),
                                 f"{m['title'] or m['path']}\n{m['kind']} · vinculado em {local_dt(m['linked_at'])}")
            it.setData(Qt.UserRole, (m["id"], m["path"]))
            self.media.addItem(it)
        self.history.clear()
        for e in d["events"]:
            self.history.addItem(f"{local_dt(e['at'])} · {EVENT_LABEL.get(e['kind'], e['kind'])}"
                                 + (f"\n{e['detail']}" if e["detail"] else ""))
        self.btn_dnc.setText("  Permitir contato novamente" if dnc else "  Não contatar")
        self.btn_cnpj.setVisible(bool(d["cnpj"]))
        self.btn_google.setVisible(any(r["source"] == "google_places" for r in d["source_records"]))
        self.btn_site.setEnabled(bool(d["domain"]))
        self._loading = False

    def _facts(self, d: dict) -> None:
        while self.facts.count():
            w = self.facts.takeAt(0).widget()
            if w:
                w.deleteLater()
        origin = d.get("origin") or {}
        rows = []
        p1, p2 = ph.normalize(d["phone_e164"] or d["phone"]), ph.normalize(d["phone2"])
        rows.append(("Endereço", d["address"], "address"))
        rows.append(("Cidade", " / ".join(x for x in (d["city"], d["uf"]) if x) + (f" · CEP {d['cep']}" if d["cep"] else ""),
                     "city"))
        rows.append(("Telefone", (p1.national + f" ({p1.kind})") if p1.valid else d["phone"], "phone"))
        if d["phone2"]:
            rows.append(("Telefone 2", (p2.national + f" ({p2.kind})") if p2.valid else d["phone2"], "phone2"))
        if d["wa_number"]:
            rows.append(("WhatsApp", f"{ph.normalize(d['wa_number']).national} · {WA_LABEL.get(d['wa_status'], '')}",
                         "whatsapp"))
        rows.append(("Site", d["website"], "website"))
        rows.append(("CNPJ", format_cnpj(d["cnpj"]) if d["cnpj"] else "", "cnpj"))
        if d["situacao"]:
            rows.append(("Situação", d["situacao"], "situacao"))
        if d["cnae"]:
            from ..keywords import CNAE
            rows.append(("CNAE", f"{d['cnae']} {CNAE.get(d['cnae'][:7], '')}".strip(), "cnae"))
        for i, (name, value, key) in enumerate(rows):
            n = style.label(name.upper(), "lfFieldName")
            v = QLabel(value or "—"); v.setObjectName("lfFieldValue"); v.setWordWrap(True)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if key == "website" and value:
                url = value if "://" in value else "https://" + value
                v.setText(f'<a style="color:{theme.ACCENT_HI}" href="{url}">{value}</a>')
                v.setTextFormat(Qt.RichText); v.setOpenExternalLinks(False)
                v.linkActivated.connect(self.ctx.open_url)
            src = origin.get(key)
            s = style.label(TITLES.get(src, src or ""), "lfSource")
            self.facts.addWidget(n, i, 0, Qt.AlignTop); self.facts.addWidget(v, i, 1); self.facts.addWidget(s, i, 2,
                                                                                                          Qt.AlignTop)
        self.facts.setColumnStretch(1, 1)
        recs = []
        for r in d["source_records"]:
            line = f"{TITLES.get(r['source'], r['source'])} · {short_date(r['fetched_at'])}"
            if r["query"]:
                line += f" · {r['query']}"
            if r["purged"]:
                line += " · dados apagados pelo prazo da licença"
            elif r["expires_at"]:
                line += f" · guardado até {short_date(r['expires_at'])}"
            recs.append(line)
        self.sources_label.setText("Origem: " + " | ".join(recs) if recs else "")

    def _score(self, d: dict) -> None:
        self.score_bar.set_value(d["score"])
        if d["score"] is None:
            self.score_label.setText("Sem pontuação")
            self.criteria.setText("")
            return
        self.score_label.setText(f"{d['score']} / 100 · {PRIORITY_LABEL[d['priority']]}")
        try:
            items = json.loads(d["score_detail"] or "[]")
        except ValueError:
            items = []
        lines = []
        for it in items:
            pts = it.get("pontos", 0)
            col = theme.OK if pts > 0 else (theme.MUTED if it.get("avaliado", True) else theme.DISABLED)
            ev = f' <span style="color:{theme.MUTED}">— {it["evidencia"]}</span>' if it.get("evidencia") else ""
            lines.append(f'<p style="margin:0 0 6px 0"><b style="color:{col}">{"+" if pts > 0 else ""}{pts}</b> '
                         f'<b>{it["criterio"]}</b>: {it["justificativa"]}{ev}</p>')
        self.criteria.setText("".join(lines))

    # ------------------------------------------------------------------------------------ ações
    def _call(self, fn, ok_msg: str = "", reload_all: bool = True) -> None:
        def done(_r):
            if ok_msg:
                self.ctx.status(ok_msg)
            if reload_all:
                self.ctx.leads_changed.emit()
            self.reload()

        def fail(e):
            QMessageBox.warning(self, "Lead Finder", friendly_error(e))
            self.reload()
        self.ctx.db.call(fn, on_done=done, on_error=fail)

    def _status_changed(self) -> None:
        if self._loading or not self.lead:
            return
        new = self.status.currentData()
        if new == self.lead["status"]:
            return
        note = ""
        if new == DO_NOT_CONTACT or self.lead["status"] == DO_NOT_CONTACT:
            if not self._confirm_dnc(new == DO_NOT_CONTACT):
                self._loading = True; self.status.setCurrentIndex(self.status.findData(self.lead["status"]))
                self._loading = False
                return
            note = "alterado manualmente"
        lid = self.lead["id"]
        self._call(lambda s: s.set_status(lid, new, note), f"Status: {STATUS_LABEL[new]}")

    def _confirm_dnc(self, setting: bool) -> bool:
        text = ("Registrar pedido de não contato? O WhatsApp e os lembretes ficam bloqueados e a empresa não volta "
                "como contatável em novas pesquisas." if setting else
                "Remover o pedido de não contato? Faça isso só se a própria empresa autorizou novo contato.")
        box = QMessageBox(QMessageBox.Question, "Não contatar", text, QMessageBox.Yes | QMessageBox.No, self)
        return box.exec() == QMessageBox.Yes

    def _toggle_dnc(self) -> None:
        if not self.lead:
            return
        if self.lead["status"] == DO_NOT_CONTACT:
            self.status.setCurrentIndex(self.status.findData("sem_interesse"))
        else:
            self.status.setCurrentIndex(self.status.findData(DO_NOT_CONTACT))

    def _priority_changed(self) -> None:
        if self._loading or not self.lead:
            return
        lid, v = self.lead["id"], self.priority.currentData()
        self._call(lambda s: s.set_priority(lid, v))

    def _wa_changed(self) -> None:
        if self._loading or not self.lead:
            return
        lid, v = self.lead["id"], self.wa_state.currentData()
        if v != WA_CONFIRMED:
            self._call(lambda s: s.set_whatsapp(lid, v), "WhatsApp atualizado.")
            return
        # confirmar: a pessoa escolhe QUAL número tem WhatsApp quando a empresa tem mais de um
        self.ctx.db.call(lambda s: s.whatsapp_candidates(lid), on_done=lambda c: self._confirm_number(lid, c))

    def _confirm_number(self, lid: int, candidates: list[str]) -> None:
        if not self.lead or self.lead["id"] != lid:
            return
        if not candidates:
            QMessageBox.warning(self, "WhatsApp", "A empresa não tem telefone válido para confirmar.")
            return self.reload()
        number = candidates[0]
        if len(candidates) > 1:
            labels = [ph.normalize(c).national + ("  (atual)" if i == 0 and self.lead["wa_number"] == c else "")
                      for i, c in enumerate(candidates)]
            choice, ok = QInputDialog.getItem(self, "Confirmar WhatsApp", "Qual número tem WhatsApp?", labels, 0, False)
            if not ok:
                return self.reload()
            number = candidates[labels.index(choice)]
        self._call(lambda s: s.set_whatsapp(lid, WA_CONFIRMED, number),
                   f"WhatsApp confirmado: {ph.normalize(number).national}.")

    def _event(self, kind: str) -> None:
        if not self.lead:
            return
        lid = self.lead["id"]
        self._call(lambda s: s.add_event(lid, kind), EVENT_LABEL[kind] + ".")

    def _add_note(self) -> None:
        text = self.note_edit.toPlainText().strip()
        if not text or not self.lead:
            return
        lid = self.lead["id"]
        self.note_edit.clear()
        self._call(lambda s: s.add_note(lid, text), "Nota adicionada.", reload_all=False)

    def _add_reminder(self) -> None:
        if not self.lead:
            return
        lid = self.lead["id"]
        when = self.rem_when.dateTime().toUTC().toString("yyyy-MM-ddTHH:mm:ss") + "+00:00"
        text = self.rem_text.text().strip()
        self.rem_text.clear()
        self._call(lambda s: s.add_reminder(lid, when, text), "Lembrete criado.")

    def _complete_reminder(self, item) -> None:
        rid = item.data(Qt.UserRole)
        self._call(lambda s: s.complete_reminder(rid), "Lembrete concluído.")

    def _rescore(self) -> None:
        from .. import ops
        lid = self.lead["id"]
        self._call(lambda s: ops.rescore(s, [lid]), "Pontuação recalculada.")

    def _copy_phone(self) -> None:
        if self.lead:
            info = ph.normalize(self.lead["wa_number"] or self.lead["phone_e164"])
            QGuiApplication.clipboard().setText(info.national if info.valid else self.lead["phone"])
            self.ctx.status("Telefone copiado.")

    def _open_media(self, item) -> None:
        _id, path = item.data(Qt.UserRole)
        self.ctx.open_url(QUrl.fromLocalFile(path).toString())

    def _unlink(self) -> None:
        it = self.media.currentItem()
        if it is None:
            return
        link_id, _ = it.data(Qt.UserRole)
        self._call(lambda s: s.unlink_media(link_id), "Vínculo removido.")

    def _delete(self) -> None:
        if not self.lead:
            return
        box = QMessageBox(QMessageBox.Warning, "Excluir dados",
                          f"Excluir “{self.lead['name']}” e todo o histórico deste computador?\n\n"
                          "Escolha “Sim” para excluir e também impedir que volte em pesquisas (pedido de não contato).",
                          QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, self)
        box.button(QMessageBox.Yes).setText("Excluir e não contatar")
        box.button(QMessageBox.No).setText("Só excluir")
        r = box.exec()
        if r == QMessageBox.Cancel:
            return
        lid, dnc = self.lead["id"], r == QMessageBox.Yes

        def done(_r):
            self.ctx.status("Dados excluídos.")
            self.load(None)
            self.ctx.leads_changed.emit()
        self.ctx.db.call(lambda s: s.delete_lead(lid, dnc), on_done=done)
